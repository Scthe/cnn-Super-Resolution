#include <iostream>
#include <memory>  // for std:unique_ptr
#include <vector>
#include <random>  // for std::mt19937
#include <chrono>  // for random seed

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"
#include "../src/LayerData.hpp"
#include "../src/LayerExecutor.hpp"
#include "TestRunner.hpp"
#include "TestDataProvider.hpp"

using namespace test::data;

///
/// ExtractLumaTest
///
DEFINE_TEST_STR(ExtractLumaTest, "Extract luma test", context) {
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

  auto kernel = _context->create_kernel("src/kernel/extract_luma.cl");
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
    assert_equals((*layer_1_input)[i], cpu_buf[i]);
  }

  return true;
}

void init(const std::vector<float> *layer_1_input) {
  this->layer_1_input = layer_1_input;
}

private:
const std::vector<float> *layer_1_input = nullptr;

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

  // create kernel
  std::stringstream kernel_compile_opts;
  if (data->result_multiply) {
    kernel_compile_opts << "-D RESULT_MULTIPLY=" << data->result_multiply;
    std::cout << "RESULT_MULTIPLY=" << data->result_multiply << " (last layer)"
              << std::endl;
  } else {
    kernel_compile_opts << "-D CURRENT_FILTER_COUNT="
                        << data->current_filter_count;
    std::cout << "CURRENT_FILTER_COUNT=" << data->current_filter_count
              << " (layers 1,2)" << std::endl;
  }
  auto kernel = _context->create_kernel("src/kernel/layer_uber_kernel.cl",
                                        kernel_compile_opts.str().c_str());

  // pre run fixes
  if (data->preproces_mean) {
    float input_mean = mean(&data->input[0], data->input_w * data->input_h);
    for (size_t i = 0; i < data->input_w * data->input_h; i++) {
      data->input[i] -= input_mean;
    }
  }

  // run
  opencl::MemoryHandler *gpu_buf_out;
  cnn_sr::LayerExecutor exec;
  cl_event finish_token = exec(*kernel, layer_data, data->input, gpu_buf_out,
                               data->input_w, data->input_h);

  const size_t out_w = data->input_w - data->f_spatial_size + 1,
               out_h = data->input_h - data->f_spatial_size + 1,
               out_count = out_w * out_h * data->current_filter_count;

  // read results
  std::unique_ptr<float[]> cpu_buf(new float[out_count]);
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_float) * out_count,
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

void init(LayerData *data) { this->data = data; }

private:
LayerData *data = nullptr;
END_TEST

///
/// SumSquaredTest
///
DEFINE_TEST_STR(SumSquaredTest, "Mean squared error - sum squared", context) {
  // total padding (from both sides) = padding*2
  const size_t algo_w = 1000, algo_h = 2000, padding = 4;  // SETTINGS
  const size_t ground_truth_w = algo_w + padding * 2,
               algo_size = algo_w * algo_h,
               ground_truth_len =
                   (algo_w + padding * 2) * (algo_h + padding * 2);
#define DBG(X) std::cout << #X << "=" << X << std::endl;
  DBG(ground_truth_w)
  DBG(algo_size)
  DBG(ground_truth_len)
#undef DBG

  std::unique_ptr<float[]> cpu_algo_res(new float[algo_size]);
  std::unique_ptr<float[]> cpu_ground_truth(new float[ground_truth_len]);
  for (size_t i = 0; i < ground_truth_len; i++) {
    cpu_ground_truth[i] = 99999.0f;
  }

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);
  double expected = 0.0;
  for (size_t i = 0; i < algo_size; i++) {
    size_t row = i / algo_w, col = i % algo_w,
           g_t_idx = (row + padding) * ground_truth_w + padding + col;
    cpu_ground_truth[g_t_idx] = generator() % 256;
    cpu_algo_res[i] = (generator() % 2560) / 10.0f;
    double d = cpu_ground_truth[g_t_idx] - cpu_algo_res[i];
    expected += d * d;
    // std::cout << d << "\t\t**2 => " << expected << std::endl;
    // std::cout << cpu_algo_res[i] << std::endl;
  }
  /*
  for (size_t i = 0; i < algo_h + padding * 2; i++) {
    size_t idx = i * ground_truth_w;
    std::cout << "[ ";
    for (size_t j = 0; j < ground_truth_w; j++) {
      std::cout << cpu_ground_truth[idx + j] << ", ";
    }
    std::cout << "]" << std::endl;
  }*/

  /* clang-format off */
  auto gpu_buf_org_img = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ground_truth_len);
  _context->write_buffer(gpu_buf_org_img, (void *)&cpu_ground_truth[0], true);
  auto gpu_buf_algo_res = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * algo_size);
  _context->write_buffer(gpu_buf_algo_res, (void *)&cpu_algo_res[0], true);

  const unsigned __int64 out_init_val = 0;
  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  _context->write_buffer(gpu_buf_out, (void *)&out_init_val, true);
  /* clang-format on */

  // kernel+args
  auto kernel = _context->create_kernel("src/kernel/sum_squared.cl");
  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*kernel, global_work_size, local_work_size, algo_w,
                            algo_h);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  kernel->push_arg(gpu_buf_org_img);
  kernel->push_arg(gpu_buf_algo_res);
  kernel->push_arg(sizeof(cl_float) * local_work_size[0], nullptr);  // scrath
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  kernel->push_arg(sizeof(cl_uint), (void *)&algo_w);
  kernel->push_arg(sizeof(cl_uint), (void *)&algo_size);

  // run
  cl_event finish_token = kernel->execute(1, global_work_size, local_work_size);

  // read (values may not be exactly the same since float->long data loss,
  // but should be close enough)
  unsigned __int64 read_val;
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_ulong), (void *)&read_val,
                        true, &finish_token, 1);
  // std::cout << "expected: " << expected << "\tgot: " << read_val <<
  // std::endl;
  assert_equals(expected, read_val);

  return true;
}

void init() {}

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

  std::vector<TestCase *> cases;
  std::vector<int> results;

  TestDataProvider data_provider;
  auto status = data_provider.read("test/data/test_cases.json");
  if (!status) {
    exit(EXIT_FAILURE);
  }

  //
  //
  //

  ADD_TEST(ExtractLumaTest, &data_provider.layer1_data.input);
  ADD_TEST(LayerTest, &data_provider.layer1_data);
  ADD_TEST(LayerTest, &data_provider.layer2_data_set1);
  ADD_TEST(LayerTest, &data_provider.layer2_data_set2);
  ADD_TEST(LayerTest, &data_provider.layer3_data);
  ADD_TEST(SumSquaredTest);

  //
  //
  //
  //

  opencl::Context context(argc, argv);
  context.init();

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
      std::cout << ex.what() << std::endl;
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
