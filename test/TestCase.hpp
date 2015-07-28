#ifndef TEST_CASE_H
#define TEST_CASE_H

#include "../src/pch.hpp"
#include <stdexcept>
#include <sstream>

namespace test {

///
/// utils functions
///

float activation_function(float);
float activation_function_derivative(float);

///
///  TestException
///
class TestException : public std::runtime_error {
 public:
  TestException();
  TestException(const char *);
  TestException(const TestException &);

  virtual const char *what() const throw();

 private:
  std::ostringstream cnvt;
};

///
/// TestCase etc.
///
struct DataSet {
  DataSet(std::string name) : name(name) {}
  DataSet() {}
  std::string name;
};

class TestCase {
 public:
  ~TestCase() {}

  virtual std::string name(size_t data_set_id) = 0;
  virtual bool operator()(size_t data_set_id, cnn_sr::DataPipeline *const) = 0;
  virtual size_t data_set_count() { return 1; }

 protected:
  void assert_equals(int expected, int result);
  void assert_equals(float expected, float result);
  void assert_equals(const std::vector<float> &expected,
                     const std::vector<float> &result, bool print = false);
  void assert_equals(cnn_sr::DataPipeline *const,
                     const std::vector<float> &expected, opencl::MemoryHandle,
                     bool print = false);
  void assert_true(bool v, const char *msg);
  void assert_data_set_ok(size_t);

  template <typename T>
  void assert_not_null(T *, const char *msg = nullptr);
};

///
/// template implementations
///

template <typename T>
void TestCase::assert_not_null(T *ptr, const char *msg) {
  if (!msg) msg = "Null pointer";
  assert_true(ptr != nullptr, msg);
}
}

#endif /* TEST_CASE_H   */
