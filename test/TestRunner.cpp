#include <iostream>
#include <vector>
#include <cmath>   // std::exp
#include <cstdio>  // snprintf

#include "../src/opencl/Context.hpp"
#include "../src/DataPipeline.hpp"
#include "TestRunner.hpp"
#include "specs/TestSpecsDeclarations.hpp"
#include "TestException.hpp"

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
  snprintf(msg_buffer, sizeof(msg_buffer),  //
           "Incorrect data set index(%d), there are only %d data sets", idx,
           this->data_set_count());
  assert_true(idx < this->data_set_count(), msg_buffer);
}
}

///
/// Test runner main function
///

#define ADD_TEST(test_name, ...)                   \
  test_name CONCATENATE(__test, __LINE__){};       \
  CONCATENATE(__test, __LINE__).init(__VA_ARGS__); \
  cases.push_back(&CONCATENATE(__test, __LINE__));

int main(int argc, char **argv) {
  std::cout << "STARTING TESTS" << std::endl;

  using namespace test;
  using namespace test::specs;

  std::vector<TestCase *> cases;
  std::vector<int> results;

  opencl::Context context;
  context.init();
  cnn_sr::DataPipeline pipeline(&context);
  pipeline.init(cnn_sr::DataPipeline::LOAD_KERNEL_MISC);

  //
  //
  //

  ADD_TEST(LayerTest);
  ADD_TEST(ExtractLumaTest);
  ADD_TEST(SwapLumaTest);
  ADD_TEST(SquaredErrorTest);
  ADD_TEST(SubtractFromAllTest);
  ADD_TEST(SumTest);
  ADD_TEST(LayerDeltasTest);
  ADD_TEST(BackpropagationTest);
  ADD_TEST(LastLayerDeltaTest);
  ADD_TEST(WeightDecayTest);
  ADD_TEST(UpdateParametersTest);
  ADD_TEST(ConfigTest);

  //
  //
  //
  //

  int failures = 0;
  for (auto i = begin(cases); i != end(cases); ++i) {
    TestCase *test = *i;
    size_t data_set_cnt = test->data_set_count();
    if (data_set_cnt == 0) {
      data_set_cnt = 1;
    }

    // run test case with all data sets
    for (size_t ds = 0; ds < data_set_cnt; ds++) {
      auto test_name = test->name(ds);
      bool passed = false;

      std::cout << std::endl
                << test_name << ":" << std::endl;
      try {
        passed = (*test)(ds, &pipeline);
      } catch (const std::exception &ex) {
        std::cout << "[ERROR] " << ex.what() << std::endl;
      } catch (...) {
        std::cout << "[ERROR] Undefined exception" << std::endl;
      }
      results.push_back(passed ? 1 : 0);
    }
  }

  // print results
  std::cout << std::endl
            << "RESULTS:" << std::endl;
  size_t test_case_it = 0;
  for (size_t i = 0; i < cases.size(); i++) {
    TestCase *test = cases[i];
    size_t data_set_cnt = test->data_set_count();
    if (data_set_cnt == 0) {
      data_set_cnt = 1;
    }
    for (size_t ds = 0; ds < data_set_cnt; ds++) {
      auto test_name = test->name(ds);
      bool passed = results[test_case_it] != 0;
      ++test_case_it;
      if (passed) {
        std::cout << "\t  " << test_name << std::endl;
      } else {
        std::cout << "\t~ " << test_name << std::endl;
        ++failures;
      }
    }
  }

  if (failures == 0) {
    std::cout << results.size() << " tests completed" << std::endl;
    exit(EXIT_SUCCESS);
  } else {
    std::cout << failures << " of " << results.size() << " failed" << std::endl;
    exit(EXIT_FAILURE);
  }
}
