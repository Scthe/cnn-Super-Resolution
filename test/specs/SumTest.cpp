#include "TestSpecsDeclarations.hpp"

#include <cstdio>  // snprintf

#include "../TestException.hpp"
#include "../../src/DataPipeline.hpp"

namespace test {
namespace specs {

///
/// PIMPL
///
struct SumTestImpl {};

///
/// SumTest
///

TEST_SPEC_PIMPL(SumTest)

void SumTest::init() {}

std::string SumTest::name(size_t sq) {
  return sq == 1 ? "Sum all test - squared" : "Sum all test";
}

size_t SumTest::data_set_count() { return 2; }

bool SumTest::operator()(size_t sq, cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto _context = pipeline->context();

  bool squared = sq == 1;
  const size_t data_len = 900;
  long long expected = 0;
  float cpu_data[data_len];
  for (size_t i = 0; i < data_len; i++) {
    cpu_data[i] = i;
    expected += squared ? i * i : i;
  }
  // std::cout << sq << "->" << squared << " exp: " << expected << std::endl;
  auto gpu_buf_data =
      _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data_len);
  _context->write_buffer(gpu_buf_data, (void *)cpu_data, true);

  cl_ulong result = pipeline->sum(gpu_buf_data, squared);

  // ok, we do not expect 100% correct result
  long long margin = 20;
  long long err = expected - result;
  err = err < 0 ? -err : err;
  if (err > margin) {
    char msg_buffer[128];
    snprintf(msg_buffer, sizeof(msg_buffer),  //
             "Expected %lld to be %lld", result, expected);
    throw TestException(msg_buffer);
  }

  return true;
}

//
//
}  // namespace specs
}  // namespace test
