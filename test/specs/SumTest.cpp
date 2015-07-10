#include "TestSpecsDeclarations.hpp"

#include "../../src/opencl/UtilsOpenCL.hpp"
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

std::string SumTest::name(size_t) { return "Sum all test"; }

size_t SumTest::data_set_count() { return 1; }

bool SumTest::operator()(size_t, cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto _context = pipeline->context();

  const size_t data_len = 900, expected = 404550;
  float cpu_data[data_len];
  for (size_t i = 0; i < data_len; i++) {
    cpu_data[i] = i;
  }
  auto gpu_buf_data =
      _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data_len);
  _context->write_buffer(gpu_buf_data, (void *)cpu_data, true);

  cl_ulong result = pipeline->sum(gpu_buf_data);
  // std::cout << result << std::endl;
  assert_equals(expected, result);

  return true;
}

//
//
}  // namespace specs
}  // namespace test
