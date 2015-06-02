#include <iostream>
#include <memory>  // for std:unique_ptr
#include <cstdio>
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
  cl_event execute_kernel(Kernel *const kernel) {
    size_t global_work_size[2] = {512, 512};
    size_t local_work_size[2] = {32, 32};
    return kernel->execute(2, global_work_size, local_work_size);
  }

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
  kernel->push_arg(sizeof(cl_mem), (void *)&gpu_image->handle);
  kernel->push_arg(sizeof(cl_mem), (void *)&gpu_buf->handle);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.h);

  // cl_event finish_token = execute_kernel(kernel);
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

///
/// Test runner main function
///
int main(int argc, char **argv) {
  std::cout << "STARTING TESTS" << std::endl;

  std::vector<TestCase *> cases;
  std::vector<int> results;
  ExtractLumaTest c{};
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
