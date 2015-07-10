#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

///
/// This file contains various definitions to make tests more concise
///

#include <string>

///
/// macro - utils
///
#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

namespace cnn_sr {
class DataPipeline;
}

namespace test {

///
/// utils functions
///
float sigmoid(float);

float mean(float *, size_t);

struct DataSet {
  DataSet(std::string name) : name(name) {}
  DataSet() {}
  std::string name;
};

/**
 * main test class to inherit from
 */
class TestCase {
 public:
  ~TestCase() {}

  virtual std::string name(size_t data_set_id) = 0;
  virtual bool operator()(size_t data_set_id, cnn_sr::DataPipeline *const) = 0;
  virtual size_t data_set_count() { return 1; }

 protected:
  void assert_equals(float expected, float result);
  void assert_true(bool v, const char *msg);
  void assert_data_set_ok(size_t);

  template <typename T>
  void assert_not_null(T *, const char *msg = nullptr);

 private:
  char msg_buffer[255];
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

#endif /* TEST_RUNNER_H   */
