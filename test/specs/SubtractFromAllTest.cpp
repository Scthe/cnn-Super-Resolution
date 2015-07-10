#include "TestSpecsDeclarations.hpp"

#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"

namespace test {
namespace specs {

///
/// PIMPL
///
struct SubtractFromAllTestImpl {};

///
/// SubtractFromAllTest
///

TEST_SPEC_PIMPL(SubtractFromAllTest)

void SubtractFromAllTest::init() {}

std::string SubtractFromAllTest::name(size_t) {
  return "Subtract from all test";
}

size_t SubtractFromAllTest::data_set_count() { return 1; }

bool SubtractFromAllTest::operator()(size_t,
                                     cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto _context = pipeline->context();

  const size_t data_len = 900;
  const float to_subtract = 450.0f;
  std::vector<float> cpu_data(data_len);
  std::vector<float> expected_buf(data_len);
  for (size_t i = 0; i < data_len; i++) {
    cpu_data[i] = i;
    expected_buf[i] = cpu_data[i] - to_subtract;
  }

  // gpu allocate
  auto gpu_buf_data =
      _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data_len);
  _context->write_buffer(gpu_buf_data, (void *)&cpu_data[0], true);

  // run
  auto finish_token = pipeline->subtract_from_all(gpu_buf_data, to_subtract);

  // read results
  _context->read_buffer(gpu_buf_data, (void *)&cpu_data[0], true, &finish_token,
                        1);

  // compare results
  for (size_t i = 0; i < data_len; i++) {
    float expected = expected_buf[i];
    float result = cpu_data[i];  // straight from gpu
    // std::cout << (i + 1) << "  expected: " << expected << "\tgot: " << result
    // << std::endl;
    assert_equals(expected, result);
  }

  return true;
}

//
//
}  // namespace specs
}  // namespace test
