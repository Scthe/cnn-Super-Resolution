#include <iostream>
#include <memory>  // for std:unique_ptr
#include <vector>

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"
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

  cl_image_format pixel_format;
  pixel_format.image_channel_order = CL_RGBA;
  pixel_format.image_channel_data_type = CL_UNSIGNED_INT8;
  auto gpu_image = _context->create_image(CL_MEM_READ_WRITE,  //
                                          data.w, data.h, &pixel_format);
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
  const size_t out_w = data->input_w - data->f_spatial_size + 1,
               out_h = data->input_h - data->f_spatial_size + 1,
               out_count = out_w * out_h * data->current_filter_count,
               input_size =
                   data->input_w * data->input_h * data->n_prev_filter_cnt;
  this->assert_true(
      data->input.size() >= input_size,
      "Declared input_w*input_h*n_prev_filter_cnt is bigger then input array");
  std::cout << "out size:" << out_w << "x" << out_h << std::endl;

  if (data->preproces_mean) {
    float input_mean = mean(&data->input[0], data->input_w * data->input_h);
    for (size_t i = 0; i < data->input_w * data->input_h; i++) {
      data->input[i] -= input_mean;
    }
  }

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_in = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * input_size);
  _context->write_buffer(gpu_buf_in, (void *)&data->input[0], true);
  auto gpu_buf_W = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data->weights.size());
  _context->write_buffer(gpu_buf_W, (void *)&data->weights[0], true);
  auto gpu_buf_B = _context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * data->bias.size());
  _context->write_buffer(gpu_buf_B, (void *)&data->bias[0], true);

  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * out_count);
  _context->zeros_float(gpu_buf_out, true);
  /* clang-format on */

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

  // args
  kernel->push_arg(gpu_buf_in);
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(gpu_buf_W);
  kernel->push_arg(gpu_buf_B);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->n_prev_filter_cnt);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->f_spatial_size);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->input_w);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->input_h);

  // run
  size_t global_work_size[2] = {16, 16};
  size_t local_work_size[2] = {8, 8};
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  // read results
  std::unique_ptr<float[]> cpu_buf(new float[out_count]);
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_float) * out_count,
                        (void *)cpu_buf.get(), true, &finish_token, 1);

  // compare results
  for (size_t i = 0; i < out_count; i++) {
    float expected = data->output[i];
    float result = cpu_buf[i];  // straight from gpu
    // std::cout << (i + 1) << "  exp: " << expected << "\tgot:" << result
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
