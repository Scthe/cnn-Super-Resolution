#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

///
/// This file contains various definitions to make tests more concise
///

#include <cmath>   // etd::exp
#include <cstdio>  // snprintf

///
/// macro - utils
///
#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

#define ABS(x) x = (x) > (-(x)) ? (x) : (-(x))

///
/// test macros
///
/** test definitions */
#define DEFINE_TEST(X, DESC, CTX_VAR)            \
  struct X : test::TestCase {                    \
    const char *name() override { return DESC; } \
  bool operator()(opencl::Context * const _##CTX_VAR) override

#define DEFINE_TEST_STR(X, STR, CTX_VAR) DEFINE_TEST(X, STR, CTX_VAR)

#define END_TEST \
  }              \
  ;

namespace test {

///
/// utils functions
///
float sigmoid(float);

float mean(float *, size_t);

///
/// main test class to inherit from
///
struct TestCase {
  virtual char const *name() = 0;
  virtual bool operator()(opencl::Context *const) = 0;

 protected:
  void assert_equals(float expected, float result);

  void assert_true(bool v, const char *msg);

 private:
  char msg_buffer[255];
};
}
#endif /* TEST_RUNNER_H   */
