#include <iostream>
#include <memory>  // for std:unique_ptr TODO just use vectors
#include <vector>
#include <random>  // for std::mt19937
#include <chrono>  // for random seed

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"
#include "../src/LayerData.hpp"
#include "../src/DataPipeline.hpp"
#include "TestRunner.hpp"
#include "TestDataProvider.hpp"
#include "specs/TestSpecsDeclarations.hpp"
#include "TestException.hpp"

using namespace test::data;
using namespace test;

namespace test {
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

float mean(float *arr, size_t count) {
  // TODO move to gpu
  float acc = 0;
  for (size_t i = 0; i < count; i++) {
    acc += arr[i];
  }
  return acc / count;
}

void TestCase::assert_equals(float expected, float result) {
  float margin = expected * 0.01f;
  ABS(margin);
  margin = margin < 0.01f ? 0.01f : margin;
  float err = expected - result;
  ABS(err);

  if (err > margin) {
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "Expected %f to be %f", result, expected);
    throw TestException<float>(expected, result, msg_buffer);
  }
}

void TestCase::assert_true(bool v, const char *msg) {
  if (!v) {
    throw TestException<float>(msg);
  }
}
}

///
/// ExtractLumaTest
///
DEFINE_TEST_STR(ExtractLumaTest, "Extract luma test", context) {
  // TODO use DataPipeline
  opencl::utils::ImageData data;
  load_image("test/data/color_grid.png", data);
  // std::cout << "img: " << data.w << "x" << data.h << "x" << data.bpp
  // << std::endl;

  this->assert_true(
      layer_1_input->size() >= (size_t)(data.w * data.h),
      "Vector of 1st layer's input values should be at least as big as test"
      " image");

  auto gpu_image = _context->create_image(CL_MEM_READ_WRITE, CL_RGBA,
                                          CL_UNSIGNED_INT8, data.w, data.h);
  _context->write_image(gpu_image, data, true);

  size_t data_total = sizeof(cl_float) * data.w * data.h;
  auto gpu_buf = _context->allocate(CL_MEM_WRITE_ONLY, data_total);

  const char *kernel_args = normalize ? "-D NORMALIZE" : "";
  auto *kernel =
      _context->create_kernel("src/kernel/extract_luma.cl", kernel_args);

  kernel->push_arg(gpu_image);
  kernel->push_arg(gpu_buf);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.h);

  size_t global_work_size[2] = {16, 16};
  size_t local_work_size[2] = {8, 8};
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  std::unique_ptr<float[]> cpu_buf(new float[data.w * data.h]);
  _context->read_buffer(gpu_buf,                               //
                        0, data_total, (void *)cpu_buf.get(),  //
                        true, &finish_token, 1);

  for (int i = 0; i < data.w * data.h; i++) {
    // std::cout << (i + 1) << ": " << cpu_buf[i] << "\t" << layer_1_input[i]
    // << std::endl;
    float expected = (*layer_1_input)[i];
    if (!normalize) expected *= 255;
    assert_equals(expected, cpu_buf[i]);
  }

  return true;
}

void init(bool normalize, const std::vector<float> *layer_1_input) {
  this->layer_1_input = layer_1_input;
  this->normalize = normalize;
}

private:
const std::vector<float> *layer_1_input = nullptr;
bool normalize;

END_TEST

///
/// LayerTest
///
DEFINE_TEST(LayerTest, data->name.c_str(), context) {
  // convert layer test definition to cnn_sr::LayerData object
  // TODO remove layer test definition as is and merge with cnn_sr::LayerData
  cnn_sr::LayerData layer_data(data->n_prev_filter_cnt,
                               data->current_filter_count, data->f_spatial_size,
                               &data->weights[0], &data->bias[0]);

  // pre run fixes
  if (data->preproces_mean) {
    float input_mean = mean(&data->input[0], data->input.size());
    for (size_t i = 0; i < data->input.size(); i++) {
      data->input[i] -= input_mean;
    }
  }

  // alloc input
  auto gpu_buf_in = _context->allocate(CL_MEM_WRITE_ONLY,
                                       sizeof(cl_float) * data->input.size());
  _context->write_buffer(gpu_buf_in, (void *)&data->input[0], true);

  // create kernel & run
  auto kernel =
      pipeline->create_layer_kernel(layer_data, data->result_multiply);
  cnn_sr::CnnLayerGpuAllocationPool gpu_alloc;
  cl_event finish_token = pipeline->execute_layer(
      *kernel, layer_data, gpu_alloc, gpu_buf_in, data->input_w, data->input_h);

  size_t out_dim[2];
  layer_data.get_output_dimensions(out_dim, data->input_w, data->input_h);
  size_t out_count = out_dim[0] * out_dim[1] * data->current_filter_count;

  // read results
  std::unique_ptr<float[]> cpu_buf(new float[out_count]);
  _context->read_buffer(gpu_alloc.output, 0, sizeof(cl_float) * out_count,
                        (void *)cpu_buf.get(), true, &finish_token, 1);

  // compare results
  for (size_t i = 0; i < out_count; i++) {
    float expected = data->output[i];
    float result = cpu_buf[i];  // straight from gpu
    // std::cout << (i + 1) << "  expected: " << expected << "\tgot: " << result
    // << std::endl;
    assert_equals(expected, result);
  }

  return true;
}

void init(cnn_sr::DataPipeline *pipeline, LayerData *data) {
  this->data = data;
  this->pipeline = pipeline;
}

private:
LayerData *data = nullptr;
cnn_sr::DataPipeline *pipeline = nullptr;
END_TEST

///
/// SumSquaredTest
///
DEFINE_TEST_STR(MeanSquaredErrorTest, "Mean squared error", context) {
  // CONSTS:
  const size_t algo_w = 1000, algo_h = 2000, padding = 4;
  // VALUES:
  // total padding (from both sides) = padding*2
  const size_t ground_truth_w = algo_w + padding * 2,
               ground_truth_h = algo_h + padding * 2,
               algo_size = algo_w * algo_h,
               ground_truth_size = ground_truth_w * ground_truth_h;

  std::unique_ptr<float[]> cpu_algo_res(new float[algo_size]);
  std::unique_ptr<float[]> cpu_expected(new float[algo_size]);
  std::unique_ptr<float[]> cpu_ground_truth(new float[ground_truth_size]);
  for (size_t i = 0; i < ground_truth_size; i++) {
    cpu_ground_truth[i] = 99999.0f;
  }

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);
  for (size_t i = 0; i < algo_size; i++) {
    size_t row = i / algo_w, col = i % algo_w,
           g_t_idx = (row + padding) * ground_truth_w + padding + col;
    cpu_ground_truth[g_t_idx] = generator() % 256;
    cpu_algo_res[i] = (generator() % 2560) / 10.0f;
    // fill expected buffer
    double d = cpu_ground_truth[g_t_idx] - cpu_algo_res[i];
    cpu_expected[i] = d * d;
  }

  /* clang-format off */
  auto gpu_buf_ground_truth = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ground_truth_size);
  _context->write_buffer(gpu_buf_ground_truth, (void *)&cpu_ground_truth[0], true);
  auto gpu_buf_algo_res = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * algo_size);
  _context->write_buffer(gpu_buf_algo_res, (void *)&cpu_algo_res[0], true);
  /* clang-format on */

  // exec
  opencl::MemoryHandler *gpu_buf_out = nullptr;
  auto finish_token = pipeline->mean_squared_error(
      gpu_buf_ground_truth, gpu_buf_algo_res, gpu_buf_out, ground_truth_w,
      ground_truth_h, padding * 2);

  // read & compare results
  std::vector<float> results(algo_size);
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_float) * algo_size,
                        (void *)&results[0], true, &finish_token, 1);
  for (size_t i = 0; i < algo_size; i++) {
    auto expected = cpu_expected[i];
    auto read_val = results[i];
    // std::cout << "expected: " << expected << "\tgot: " << read_val <<
    // std::endl;
    assert_equals(expected, read_val);
  }
  return true;
}

void init(cnn_sr::DataPipeline *pipeline) { this->pipeline = pipeline; }

private:
cnn_sr::DataPipeline *pipeline;

END_TEST

///
/// ExtractLumaTest
///
DEFINE_TEST_STR(SumTest, "Sum test", context) {
  size_t data_len = 900, expected = 404550;
  std::unique_ptr<float[]> cpu_data(new float[data_len]);
  for (size_t i = 0; i < data_len; i++) {
    cpu_data[i] = i;
  }
  auto gpu_buf_data =
      _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data_len);
  _context->write_buffer(gpu_buf_data, (void *)&cpu_data[0], true);

  cl_ulong result;
  pipeline->sum(gpu_buf_data, &result);
  // std::cout << result << std::endl;
  assert_equals(expected, result);

  return true;
}

void init(cnn_sr::DataPipeline *pipeline) { this->pipeline = pipeline; }

private:
cnn_sr::DataPipeline *pipeline;

END_TEST

///
/// SubtractFromAllTest
///
DEFINE_TEST_STR(SubtractFromAllTest, "Subtract from all test", context) {
  size_t data_len = 900;
  float to_subtract = 450.0f;
  std::unique_ptr<float[]> cpu_data(new float[data_len]);
  std::unique_ptr<float[]> expected_buf(new float[data_len]);
  for (size_t i = 0; i < data_len; i++) {
    cpu_data[i] = i;
    expected_buf[i] = cpu_data[i] - to_subtract;
  }
  auto gpu_buf_data =
      _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data_len);
  _context->write_buffer(gpu_buf_data, (void *)&cpu_data[0], true);

  auto finish_token = pipeline->subtract_from_all(gpu_buf_data, to_subtract);

  // read results
  _context->read_buffer(gpu_buf_data, 0, sizeof(cl_float) * data_len,
                        (void *)cpu_data.get(), true, &finish_token, 1);

  // compare results
  for (size_t i = 0; i < data_len; i++) {
    float expected = expected_buf[i];
    float result = cpu_data[i];  // straight from gpu
    // std::cout << (i + 1) << "  expected: " << expected << "\tgot: " << result
    // << std::endl;
    assert_equals(expected, result);
  }

  return true;
}

void init(cnn_sr::DataPipeline *pipeline) { this->pipeline = pipeline; }

private:
cnn_sr::DataPipeline *pipeline;

END_TEST

///
/// Test runner main function
///

#define ADD_TEST(test_name, ...)                   \
  test_name CONCATENATE(__test, __LINE__){};       \
  CONCATENATE(__test, __LINE__).init(__VA_ARGS__); \
  cases.push_back(&CONCATENATE(__test, __LINE__));

int main(int argc, char **argv) {
  std::cout << "STARTING TESTS" << std::endl;

  using namespace test::data;
  using namespace test::specs;

  std::vector<TestCase *> cases;
  std::vector<int> results;

  TestDataProvider data_provider;
  auto status = data_provider.read("test/data/test_cases.json");
  if (!status) {
    exit(EXIT_FAILURE);
  }

  opencl::Context context(argc, argv);
  context.init();
  cnn_sr::DataPipeline pipeline(&context);
  pipeline.init(cnn_sr::DataPipeline::LOAD_KERNEL_MISC);

  //
  //
  //

  ADD_TEST(ExtractLumaTest, false, &data_provider.layer1_data.input);
  ADD_TEST(ExtractLumaTest, true, &data_provider.layer1_data.input);
  ADD_TEST(LayerTest, &pipeline, &data_provider.layer1_data);
  ADD_TEST(LayerTest, &pipeline, &data_provider.layer2_data_set1);
  ADD_TEST(LayerTest, &pipeline, &data_provider.layer2_data_set2);
  ADD_TEST(LayerTest, &pipeline, &data_provider.layer3_data);
  ADD_TEST(SumTest, &pipeline);
  ADD_TEST(SubtractFromAllTest, &pipeline);
  ADD_TEST(MeanSquaredErrorTest, &pipeline);
  ADD_TEST(LayerDeltasTest, &pipeline);
  ADD_TEST(BackpropagationTest, &pipeline);

  //
  //
  //
  //

  int failures = 0;
  for (auto i = begin(cases); i != end(cases); ++i) {
    TestCase *test = *i;
    auto test_name = test->name();
    bool passed = false;

    std::cout << std::endl
              << test_name << ":" << std::endl;

    // run test
    try {
      passed = (*test)(&context);

    } catch (const std::exception &ex) {
      std::cout << "[ERROR] " << ex.what() << std::endl;
    } catch (...) {
      std::cout << "Undefined exception" << std::endl;
    }
    results.push_back(passed ? 1 : 0);
  }

  // print results
  std::cout << std::endl
            << "RESULTS:" << std::endl;
  for (size_t i = 0; i < cases.size(); i++) {
    auto test_name = cases[i]->name();
    bool passed = results[i] != 0;
    if (passed) {
      std::cout << "\t  " << test_name << std::endl;
    } else {
      std::cout << "\t~ " << test_name << std::endl;
      ++failures;
    }
  }

  if (failures == 0) {
    std::cout << cases.size() << " tests completed" << std::endl;
    exit(EXIT_SUCCESS);
  } else {
    std::cout << failures << " of " << cases.size() << " failed" << std::endl;
    exit(EXIT_FAILURE);
  }
}
