#include <iostream>

#include "TestCase.hpp"
#include "../src/opencl/Context.hpp"
#include "../src/DataPipeline.hpp"
#include "specs/TestSpecsDeclarations.hpp"

///
/// Test runner main function
///

#define ADD_TEST(test_name, ...)                   \
  test_name CONCATENATE(__test, __LINE__){};       \
  CONCATENATE(__test, __LINE__).init(__VA_ARGS__); \
  cases.push_back(&CONCATENATE(__test, __LINE__));

int main(int, char **) {
  std::cout << "STARTING TESTS" << std::endl;

  using namespace test;
  using namespace test::specs;

  std::vector<TestCase *> cases;
  std::vector<int> results;

  opencl::Context context;
  context.init();
  cnn_sr::DataPipeline pipeline(&context);
  pipeline.init(false, cnn_sr::DataPipeline::LOAD_KERNEL_MISC);
  // TODO test opt
  // pipeline.init(true, cnn_sr::DataPipeline::LOAD_KERNEL_MISC);

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
