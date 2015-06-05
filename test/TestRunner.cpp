#include <iostream>
#include <memory>  // for std:unique_ptr
#include <vector>

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"
#include "TestRunner.hpp"
#include "TestDataProvider.hpp"

using namespace test::data;

///
/// Test definitions
///
DEFINE_TEST(ExtractLumaTest, context) {
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

DEFINE_TEST(Layer1Test, context) {
  const int out_w = data->input_w - data->f1 + 1,
            out_h = data->input_h - data->f1 + 1,
            size_per_filter = out_w * out_h;
  const size_t pixel_count = data->input_w * data->input_h;

  this->assert_true(data->input.size() >= pixel_count,
                    "Vector of 1st layer's input values should be at least as "
                    "big as test image");

  float input_mean = mean(&data->input[0], pixel_count);
  for (size_t i = 0; i < pixel_count; i++) {
    data->input[i] -= input_mean;
    // std::cout << data->input[i] << std::endl;
  }

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_in = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * pixel_count);
  _context->write_buffer(gpu_buf_in, (void *)&data->input[0], true);
  auto gpu_buf_W = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data->weights.size());
  _context->write_buffer(gpu_buf_W, (void *)&data->weights[0], true);
  auto gpu_buf_B = _context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * data->bias.size());
  _context->write_buffer(gpu_buf_B, (void *)&data->bias[0], true);

  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * size_per_filter * data->n1);
 _context->zeros_float(gpu_buf_out, true);
 //* clang-format on */

  std::stringstream kernel_compile_opts;
  kernel_compile_opts << "-D N1_FILTER_COUNT=" << data->n1;
  auto kernel = _context->create_kernel("src/kernel/layer_1.cl",
                                        kernel_compile_opts.str().c_str());

  kernel->push_arg(gpu_buf_in);
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(gpu_buf_W);
  kernel->push_arg(gpu_buf_B);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->f1);
  // TODO int or uint ?
  kernel->push_arg(sizeof(cl_uint), (void *)&data->input_w);
  kernel->push_arg(sizeof(cl_uint), (void *)&data->input_h);

  size_t global_work_size[2] = {16, 16};
  size_t local_work_size[2] = {8, 8};
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  std::unique_ptr<float[]> cpu_buf(new float[size_per_filter * data->n1]);
  _context->read_buffer(gpu_buf_out, 0,
                        sizeof(cl_float) * size_per_filter * data->n1,
                        (void *)cpu_buf.get(), true, &finish_token, 1);

  for (int i = 0; i < size_per_filter; i++) {
    size_t base_idx = i * data->n1;
    for (size_t filter_id = 0; filter_id < data->n1; filter_id++) {
      float expected = (*layer_2_input)[base_idx + filter_id];
      float result = cpu_buf[base_idx + filter_id];  // straight from gpu
      // std::cout << (i + 1) << "  exp: " << expected << "\tgot:" << result
      // << std::endl;
      assert_equals(expected, result);
    }
  }

  return true;
}

void init(Layer1Data *data, const std::vector<float> *layer_2_input) {
  this->layer_2_input = layer_2_input;
  this->data = data;
}

private:
const std::vector<float> *layer_2_input = nullptr;
Layer1Data *data = nullptr;

END_TEST

DEFINE_TEST(Layer2Test, context) {
  const int out_w = layer_1->input_w - layer_1->f1 - layer_2->f2 + 2,
            out_h = layer_1->input_h - layer_1->f1 - layer_2->f2 + 2,
            size_per_filter = out_w * out_h;
  const size_t in_size = layer_1->f1 * layer_1->f1 * layer_1->n1;
  std::cout << "out size:" << out_w << "x" << out_h << std::endl;

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_in = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * in_size);
  _context->write_buffer(gpu_buf_in, (void *)&layer_2->input[0], true);
  auto gpu_buf_W = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * layer_2->weights.size());
  _context->write_buffer(gpu_buf_W, (void *)&layer_2->weights[0], true);
  auto gpu_buf_B = _context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * layer_2->bias.size());
  _context->write_buffer(gpu_buf_B, (void *)&layer_2->bias[0], true);

  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * size_per_filter * layer_2->n2);
  _context->zeros_float(gpu_buf_out, true);
  /* clang-format on */

  std::stringstream kernel_compile_opts;
  kernel_compile_opts << "-D N2_FILTER_COUNT=" << layer_2->n2;
  auto kernel = _context->create_kernel("src/kernel/layer_2.cl",
                                        kernel_compile_opts.str().c_str());

  kernel->push_arg(gpu_buf_in);
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(gpu_buf_W);
  kernel->push_arg(gpu_buf_B);
  kernel->push_arg(sizeof(cl_uint), (void *)&layer_1->f1);
  kernel->push_arg(sizeof(cl_uint), (void *)&layer_1->n1);
  kernel->push_arg(sizeof(cl_uint), (void *)&layer_2->f2);

  size_t global_work_size[2] = {16, 16};
  size_t local_work_size[2] = {8, 8};
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  std::unique_ptr<float[]> cpu_buf(new float[size_per_filter * layer_2->n2]);
  _context->read_buffer(gpu_buf_out, 0,
                        sizeof(cl_float) * size_per_filter * layer_2->n2,
                        (void *)cpu_buf.get(), true, &finish_token, 1);

  // compare results
  for (int i = 0; i < size_per_filter; i++) {
    size_t base_idx = i * layer_2->n2;
    for (size_t filter_id = 0; filter_id < layer_2->n2; filter_id++) {
      float expected = layer_2->output[base_idx + filter_id];
      // expected = sigmoid(expected);
      float result = cpu_buf[base_idx + filter_id];  // straight from gpu
      // std::cout << (i + 1) << "  exp: " << expected << "\tgot:" << result
      // << std::endl;
      assert_equals(expected, result);
    }
  }

  /*
  size_t exp_to_read = OVERRIDE_SIZE;
  float cpu_buf[OVERRIDE_SIZE];
  for (size_t i = 0; i < OVERRIDE_SIZE; i++) cpu_buf[i] = -999;
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_float) * exp_to_read,
                        (void *)cpu_buf, true, &finish_token, 1);
  std::cout << std::endl;
  for (size_t i = 0; i < OVERRIDE_SIZE; i++) {
    std::cout << cpu_buf[i] << ", ";
    if ((i + 1) % 3 == 0) std::cout << std::endl;
  }
  */

  return true;
}

void init(Layer1Data *layer_1, Layer2Data *layer_2) {
  this->layer_1 = layer_1;
  this->layer_2 = layer_2;
}

private:
Layer1Data *layer_1 = nullptr;
Layer2Data *layer_2 = nullptr;
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
  //

  ADD_TEST(ExtractLumaTest, &data_provider.layer1_data.input);
  ADD_TEST(Layer1Test, &data_provider.layer1_data,
           &data_provider.layer2_data.input);
  ADD_TEST(Layer2Test, &data_provider.layer1_data, &data_provider.layer2_data);

  //
  //
  //
  //

  opencl::Context context(argc, argv);
  context.init();

  int failures = 0;
  for (auto i = begin(cases); i != end(cases); ++i) {
    auto test = *i;
    auto test_name = test->name();
    bool passed = false;

    std::cout << std::endl
              << test_name << ":" << std::endl;

    // run test
    try {
      passed = (*test)(&context);

    } catch (const std::exception &ex) {
      std::cout << test_name << ":" << std::endl
                << ex.what() << std::endl;
    } catch (...) {
      std::cout << test_name << ":" << std::endl
                << "Undefined exception" << std::endl;
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
