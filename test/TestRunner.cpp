#include <iostream>
#include <memory>  // for std:unique_ptr
#include <cstdio>
#include <cmath>
#include <vector>
// #include <time.h>
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"
#include "TestException.hpp"

#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

// constexpr static char const*const NAME = STRINGIFY(x);
// char const *const NAME = STRINGIFY(x);
#define DEFINE_TEST(x, ctx_var)                  \
  struct x : TestCase {                          \
    const char *const NAME = STRINGIFY(x);       \
    const char *name() override { return NAME; } \
  bool operator()(opencl::Context * const _##ctx_var) override

#define END_TEST \
  }              \
  ;

using namespace opencl;

struct TestCase {
  virtual char const *name() = 0;
  virtual bool operator()(Context *const) = 0;

 protected:
  void assert_equals(float expected, float result) {
    float err = expected - result;
    err = err > 0 ? err : -err;
    if (err > 0.1) {
      snprintf(msg_buffer, sizeof(msg_buffer),  //
               "Expected %f to be ~%f", result, expected);
      throw TestException<float>(expected, result, msg_buffer);
    }
  }

  void assert(bool v, const char *msg) {
    if (!v) {
      throw TestException<float>(msg);
    }
  }

 private:
  char msg_buffer[255];
};

///
/// Test definitions
///
DEFINE_TEST(ExtractLumaTest, context) {
  utils::ImageData data;
  load_image("test/data/color_grid.png", data);
  // std::cout << "img: " << data.w << "x" << data.h << "x" << data.bpp
  // << std::endl;

  cl_image_format pixel_format;
  pixel_format.image_channel_order = CL_RGBA;
  pixel_format.image_channel_data_type = CL_UNSIGNED_INT8;
  auto gpu_image = _context->create_image(CL_MEM_READ_WRITE,  //
                                          data.w, data.h, &pixel_format);
  _context->write_image(gpu_image, data, true);

  size_t data_total = sizeof(cl_float) * data.w * data.h;
  auto gpu_buf = _context->allocate(CL_MEM_WRITE_ONLY, data_total, nullptr);

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
  float results[25] = {0.0f,     255.0f,   207.073f, 217.543f, 111.446f,  //
                       43.246f,  178.755f, 105.315f, 225.93f,  200.578f,  //
                       109.577f, 76.245f,  149.685f, 29.07f,   180.345f,  //
                       170.892f, 190.035f, 217.543f, 190.035f, 76.278f,   //
                       205.58f,  149.852f, 218.917f, 151.138f, 179.001f};

  for (int i = 0; i < data.w * data.h; i++) {
    // std::cout << (i + 1) << ": " << cpu_buf[i] << "\t" << results[i]
    // << std::endl;
    assert_equals(results[i], cpu_buf[i]);
  }

  return true;
}
END_TEST

float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

DEFINE_TEST(Layer1Test, context) {
  // TODO normalize by 255
  float data[25] = {0.0f,     255.0f,   207.073f, 217.543f, 111.446f,  //
                    43.246f,  178.755f, 105.315f, 225.93f,  200.578f,  //
                    109.577f, 76.245f,  149.685f, 29.07f,   180.345f,  //
                    170.892f, 190.035f, 217.543f, 190.035f, 76.278f,   //
                    205.58f,  149.852f, 218.917f, 151.138f, 179.001f};
  float W[27] = {1.0f, 0.0f, 0.0f,   // 0,0
                 0.0f, 1.0f, 0.0f,   // 1,0
                 1.0f, 0.0f, 0.0f,   // 2,0
                 0.0f, 1.0f, 0.0f,   // 0,1
                 1.0f, 0.0f, 1.0f,   // 1,1
                 0.0f, 1.0f, 0.0f,   // 2,1
                 1.0f, 0.0f, 0.0f,   // 0,2
                 0.0f, 1.0f, 0.0f,   // 1,2
                 1.0f, 0.0f, 0.0f};  // 2,2
  // float B[3] = {0.1f, 0.2f, 0.3f};
  float B[3] = {0.0f, 0.0f, 0.0f};

  const size_t n1 = 3, f1 = 3;  // change n1 in N1_FILTER_COUNT too
  const int w = 5, h = 5, out_w = w - f1 + 1, out_h = h - f1 + 1,
            size_per_filter = out_w * out_h;
  const size_t pixel_count = w * h;
  std::cout << "predicted result size: " << out_w << "x" << out_h << std::endl;

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_in = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * pixel_count, nullptr);
  _context->write_buffer(gpu_buf_in, 0, sizeof(cl_float) * pixel_count, (void *)data, true);
  auto gpu_buf_W = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * sizeof(W), nullptr);
  _context->write_buffer(gpu_buf_W, 0, sizeof(cl_float) * sizeof(W), (void *)W, true);
  auto gpu_buf_B = _context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * sizeof(B), nullptr);
  _context->write_buffer(gpu_buf_B, 0, sizeof(cl_float) * sizeof(B), (void *)B, true);

  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * size_per_filter * n1, nullptr);
  std::vector<float> tmp;
  for (size_t i = 0; i < size_per_filter * n1; i++) {
    tmp.push_back(-999.5);
  }
  _context->write_buffer(gpu_buf_out, 0, sizeof(cl_float) * size_per_filter * n1, (void *)&tmp[0], true);
  /* clang-format on */

  auto kernel =
      _context->create_kernel("src/kernel/layer_1.cl", "-D N1_FILTER_COUNT=3");
  kernel->push_arg(gpu_buf_in);
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(gpu_buf_W);
  kernel->push_arg(gpu_buf_B);
  kernel->push_arg(sizeof(cl_uint), (void *)&f1);
  kernel->push_arg(sizeof(cl_uint), (void *)&w);
  kernel->push_arg(sizeof(cl_uint), (void *)&h);

  size_t global_work_size[2] = {16, 16};
  size_t local_work_size[2] = {8, 8};
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  std::unique_ptr<float[]> cpu_buf(new float[pixel_count * n1]);
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_float) * size_per_filter * n1,
                        (void *)cpu_buf.get(), true, &finish_token, 1);

  // result without B multiply and without max
  float results[27] = {645.090f, 479.806f, 178.755f,   // 0,0
                       683.173f, 761.443f, 105.315f,   // 1,0
                       874.479f, 552.506f, 225.93f,    // 2,0
                       613.241f, 628.052f, 76.245f,    // 0,1
                       934.440f, 428.173f, 149.685f,   // 1,1
                       628.784f, 745.995f, 29.07f,     // 2,1
                       873.794f, 614.532f, 190.035f,   // 0,2
                       623.848f, 748.672f, 217.543f,   // 1,2
                       917.983f, 474.029f, 190.035f};  // 2,2

  for (int i = 0; i < size_per_filter; i++) {
    size_t row = i / f1, col = i % f1;
    size_t base_idx = ((row * out_w) + col) * n1;
    std::cout << row << ":" << col << " [";
    for (size_t filter_id = 0; filter_id < n1; filter_id++) {
      std::cout << cpu_buf[base_idx + filter_id] << ", ";
      float expected = sigmoid(results[base_idx + filter_id] + B[filter_id]);
      // float expected = results[base_idx + filter_id] + B[filter_id];
      float result = cpu_buf[base_idx + filter_id];
      assert_equals(expected, result);
    }
    std::cout << "]" << std::endl;
  }

  return true;
}
END_TEST

///
/// Test runner main function
///
int main(int argc, char **argv) {
  std::cout << "STARTING TESTS" << std::endl;

  std::vector<TestCase *> cases;
  std::vector<int> results;

  // ExtractLumaTest c{};
  // cases.push_back(&c);
  Layer1Test c{};
  cases.push_back(&c);

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
