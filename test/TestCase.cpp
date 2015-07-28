#include "TestCase.hpp"

#include <iostream>
#include <cmath>   // std::abs
#include <cstdio>  // snprintf

#include "../src/opencl/Context.hpp"
#include "../src/DataPipeline.hpp"

namespace test {

///
/// utils functions
///
float activation_function(float x) { return std::max(x, 0.0f); }
float activation_function_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

///
/// TestException
///
TestException::TestException() : runtime_error("TestException") {
  // cnvt.str("");
  cnvt << runtime_error::what() << ": Undefined error";
}

TestException::TestException(const char *msg) : runtime_error("TestException") {
  // cnvt.str("");
  cnvt << runtime_error::what() << ": " << msg;
}

TestException::TestException(const TestException &e)
    : runtime_error("TestException"), cnvt(e.cnvt.str()) {}

const char *TestException::what() const throw() { return cnvt.str().c_str(); }

///
/// TestCase
///
void TestCase::assert_equals(int expected, int result) {
  if (expected != result) {
    char msg_buffer[128];
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "[INT] Expected %d to be %d", result, expected);
    throw TestException(msg_buffer);
  }
}

void TestCase::assert_equals(float expected, float result) {
  // (yeah, this are going to be totally arbitrary numbers)
  expected = std::abs(expected);
  float margin = 0.005f;
  if (expected > 10) margin = 0.15f;
  if (expected > 100) margin = 1;
  if (expected > 1000) margin = expected / 10000;
  float err = expected - std::abs(result);

  if (err > margin) {
    char msg_buffer[128];
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "[FLOAT] Expected %f to be %f", result, expected);
    throw TestException(msg_buffer);
  }
}

void TestCase::assert_true(bool v, const char *msg) {
  if (!v) {
    throw TestException(msg);
  }
}

void TestCase::assert_equals(const std::vector<float> &expected,
                             const std::vector<float> &result, bool print) {
  if (expected.size() != result.size()) {
    char msg_buffer[128];
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "Expected vector has %d elements, while result %d. This vectors "
             "are not equal",
             expected.size(), result.size());
    throw TestException(msg_buffer);
  }

  for (size_t i = 0; i < expected.size(); i++) {
    float r = result[i];
    float e = expected[i];
    if (print)
      std::cout << "[" << i << "] expected >\t" << e << "\tgot> " << r
                << std::endl;
    assert_equals(e, r);
  }
}
void TestCase::assert_equals(cnn_sr::DataPipeline *const pipeline,
                             const std::vector<float> &expected,
                             opencl::MemoryHandle handle, bool print) {
  auto context = pipeline->context();
  auto raw_gpu_mem = context->raw_memory(handle);
  size_t len = raw_gpu_mem->size / sizeof(cl_float);
  if (expected.size() != len) {
    char msg_buffer[128];
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "Expected vector has %d elements, while gpu memory holds %d. This "
             "vectors are not equal",
             expected.size(), len);
    throw TestException(msg_buffer);
  }

  context->block();
  std::vector<float> gpu_read(len);
  context->read_buffer(handle, (void *)&gpu_read[0], true);
  assert_equals(expected, gpu_read, print);
}

void TestCase::assert_data_set_ok(size_t idx) {
  char msg_buffer[128];
  snprintf(msg_buffer, sizeof(msg_buffer),  //
           "Incorrect data set index(%d), there are only %d data sets", idx,
           this->data_set_count());
  assert_true(idx < this->data_set_count(), msg_buffer);
}
}
