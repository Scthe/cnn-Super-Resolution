#ifndef __TEST_CONST_H
#define __TEST_CONST_H

///
/// This file contains various definitions to make tests more concise
///

#include "TestException.hpp"
#include <cmath>   // etd::exp
#include <cstdio>  // snprintf

///
/// macros
///
#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

/** test definitions */
#define DEFINE_TEST(x, ctx_var)                  \
  struct x : TestCase {                          \
    const char *const NAME = STRINGIFY(x);       \
    const char *name() override { return NAME; } \
  bool operator()(opencl::Context * const _##ctx_var) override

#define END_TEST \
  }              \
  ;

///
/// utils functions
///
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

///
/// main test class to inherit from
///
struct TestCase {
  virtual char const *name() = 0;
  virtual bool operator()(opencl::Context *const) = 0;

 protected:
  void assert_equals(float expected, float result) {
    float err = expected - result;
    err = err > 0 ? err : -err;
    if (err > 0.1) {
      snprintf(msg_buffer, sizeof(msg_buffer),  //
               "Expected %f to be %f", result, expected);
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
/// data
///

namespace layer_1 {

/**
 * filter count
 */
const size_t n1 = 3;

/**
 * spatial size
 */
const size_t f1 = 3;

/**
 * size of the input
 */
const int in_w = 5, in_h = 5;

/**
 * this is output of the luma extract process that will be feed to first layer
 * TODO normalize by 255
 */
float input[25] = {0.0f,     255.0f,   207.073f, 217.543f, 111.446f,  //
                   43.246f,  178.755f, 105.315f, 225.93f,  200.578f,  //
                   109.577f, 76.245f,  149.685f, 29.07f,   180.345f,  //
                   170.892f, 190.035f, 217.543f, 190.035f, 76.278f,   //
                   205.58f,  149.852f, 218.917f, 151.138f, 179.001f};

/**
 * weights for first layer
 */
float W[27] = {1.0f, 0.0f, 0.0f,   // 0,0
               0.0f, 1.0f, 0.0f,   // 1,0
               1.0f, 0.0f, 0.0f,   // 2,0
               0.0f, 1.0f, 0.0f,   // 0,1
               1.0f, 0.0f, 1.0f,   // 1,1
               0.0f, 1.0f, 0.0f,   // 2,1
               1.0f, 0.0f, 0.0f,   // 0,2
               0.0f, 1.0f, 0.0f,   // 1,2
               1.0f, 0.0f, 0.0f};  // 2,2

/**
 * biases for first layer
 */
float B[3] = {0.1f, 0.2f, 0.3f};

/**
 * NOTE: this does not add the bias nor applies max 'squashing' !
 * (Though this form allows for easier debugging)
 */
float output_raw[27] = {645.090f, 479.806f, 178.755f,   // 0,0
                        683.173f, 761.443f, 105.315f,   // 1,0
                        874.479f, 552.506f, 225.93f,    // 2,0
                        613.241f, 628.052f, 76.245f,    // 0,1
                        934.440f, 428.173f, 149.685f,   // 1,1
                        628.784f, 745.995f, 29.07f,     // 2,1
                        873.794f, 614.532f, 190.035f,   // 0,2
                        623.848f, 748.672f, 217.543f,   // 1,2
                        917.983f, 474.029f, 190.035f};  // 2,2
}

namespace layer_2 {

/**
 * filter count
 */
const size_t n2 = 2;

/**
 * spatial size
 */
const size_t f2 = 3;

/* clang-format off */
/**
 * weights for second layer ((f2*f2*n1)*n2 dimensional)
 * in other words each of n2 filters have a f2*f2*n1 cube of weights
 */
float W[54] = {  // cube's 1st row, cell 0,0
    1.000f, 2.000f,
    1.001f, 2.001f,
    1.002f, 2.002f,
    // 0,1
    1.010f, 2.010f,
    1.011f, 2.011f,
    1.012f, 2.012f,
    // 0,2
    1.020f, 2.020f,
    1.021f, 2.021f,
    1.022f, 2.022f,
    // cube's 2nd row, cell 1,0
    1.100f, 2.100f,
    1.101f, 2.101f,
    1.102f, 2.102f,
    // 1,1
    1.110f, 2.110f,
    1.111f, 2.111f,
    1.112f, 2.112f,
    // 1,2
    1.120f, 2.120f,
    1.121f, 2.121f,
    1.122f, 2.122f,
    // cube's 3rd row, cell 2,0
    1.200f, 2.200f,
    1.201f, 2.201f,
    1.202f, 2.202f,
    // 2,1
    1.210f, 2.210f,
    1.211f, 2.211f,
    1.212f, 2.212f,
    // 2,2
    1.220f, 2.220f,
    1.221f, 2.221f,
    1.222f, 2.222f};
/* clang-format on */

/**
 * biases for second layer (n2 dimensional)
 */
float B[2] = {0.1f, 0.2f};

/**
 * Create input values. (See note to layer_1::output_raw)
 * @param arr array to fill, of size: (layer_1::f1)^2 * layer_1::n1
 */
void create_input(float *arr) {
  int n1 = layer_1::n1;
  size_t f1 = layer_1::f1;

  for (size_t i = 0; i < f1 * f1; i++) {
    size_t base_idx = i * n1;
    for (int filter_id = 0; filter_id < n1; filter_id++) {
      arr[base_idx + filter_id] = sigmoid(
          layer_1::output_raw[base_idx + filter_id] + layer_1::B[filter_id]);
    }
  }
}
}

#endif /* __TEST_CONST_H   */
