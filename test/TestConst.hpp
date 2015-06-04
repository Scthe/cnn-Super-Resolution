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

float mean(float *arr, size_t count) {
  // TODO move to gpu
  float acc = 0;
  for (size_t i = 0; i < count; i++) {
    acc += arr[i];
  }
  return acc / count;
}

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
 */
float input[25] = {0.0f,     1.0f,     0.81205f, 0.85311f, 0.43704f,
                   0.16959f, 0.701f,   0.413f,   0.886f,   0.78658f,
                   0.42971f, 0.299f,   0.587f,   0.114f,   0.70724f,
                   0.67016f, 0.74524f, 0.85311f, 0.74524f, 0.29913f,
                   0.8062f,  0.58765f, 0.8585f,  0.5927f,  0.70196f};

/**
 * weights for first layer
 * (each column for different filter)
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

float input[27] = {0.406f, 0.419f, 0.598f,   // 0,0
                   0.442f, 0.685f, 0.528f,   // 0,1
                   0.627f, 0.489f, 0.642f,   // 0,2
                   0.376f, 0.563f, 0.499f,   // 1,0
                   0.680f, 0.371f, 0.571f,   // 1,1
                   0.390f, 0.672f, 0.453f,   // 1,2
                   0.626f, 0.550f, 0.609f,   // 2,0
                   0.386f, 0.674f, 0.634f,   // 2,1
                   0.666f, 0.413f, 0.609f};  // 2,2

float output[2] = {16.3343f, 31.0135};

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
}

#endif /* __TEST_CONST_H   */
