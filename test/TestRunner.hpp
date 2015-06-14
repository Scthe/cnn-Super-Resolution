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
#define DEFINE_TEST(X, DESC, CTX_VAR)              \
  struct X : TestCase {                            \
    const char *name() override { return DESC; }   \
  bool operator()(opencl::Context * const _##CTX_VAR) override

#define DEFINE_TEST_STR(X, STR, CTX_VAR) DEFINE_TEST(X, STR, CTX_VAR)

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
#define ABS(x) x = (x) > 0 ? (x) : (-(x))
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
#undef ABS
  }

  void assert_true(bool v, const char *msg) {
    if (!v) {
      throw TestException<float>(msg);
    }
  }

 private:
  char msg_buffer[255];
};

#endif /* __TEST_CONST_H   */
