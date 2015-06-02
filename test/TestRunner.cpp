#include <iostream>
// #include <time.h>
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>

#include "../src/opencl/Context.hpp"
#include "../src/opencl/UtilsOpenCL.hpp"

#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

// constexpr static char const*const NAME = STRINGIFY(x);
// char const *const NAME = STRINGIFY(x);
#define DEFINE_TEST(x, ctx_var)                  \
  struct x : TestCase {                          \
    const char *const NAME = STRINGIFY(x);       \
    const char *name() override { return NAME; } \
  bool operator()(opencl::Context * const ctx_var) override

#define END_TEST \
  }              \
  ;

struct TestCase {
  virtual char const *name() = 0;
  virtual bool operator()(opencl::Context *const) = 0;
};

DEFINE_TEST(ExtractLumaTest, context) {
  //
  return false;
}
END_TEST

DEFINE_TEST(ExtractLumaTest2, context) {
  //
  return true;
}
END_TEST

int main(int argc, char **argv) {
  std::cout << "STARTING TESTS" << std::endl;

  std::vector<TestCase *> cases;
  ExtractLumaTest c{};
  cases.push_back(&c);
  ExtractLumaTest2 c2{};
  cases.push_back(&c2);

  cases.push_back(&c2);
  cases.push_back(&c);
  cases.push_back(&c);
  cases.push_back(&c2);
  cases.push_back(&c2);

  int failures = 0;
  for (auto i = begin(cases); i != end(cases); ++i) {
    auto test = *i;
    auto test_name = test->name();
    bool passed = false;

    // run test
    try {
      // passed = test(context);
      passed = (*test)(nullptr);
    } catch (const std::exception &ex) {
      std::cout << test_name << ":" << std::endl
                << ex.what() << std::endl;
    } catch (...) {
      std::cout << test_name << ":" << std::endl
                << "Undefined exception" << std::endl;
    }

    if (!passed) {
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
