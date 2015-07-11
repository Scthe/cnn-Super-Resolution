#include "TestSpecsDeclarations.hpp"

#include <random>  // for std::mt19937
#include <chrono>  // for random seed
#
#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"

namespace test {
namespace specs {

///
/// PIMPL
///
struct WeightDecayTestImpl {
  const size_t w1_size = 2555, w2_size = 4000, w3_size = 30;
  const float weight_decay_parameter = 0.354f;
};

///
/// WeightDecayTest
///

TEST_SPEC_PIMPL(WeightDecayTest)

void WeightDecayTest::init() {}

std::string WeightDecayTest::name(size_t) { return "Weight decay test"; }

size_t WeightDecayTest::data_set_count() { return 1; }

float fill_rand(std::mt19937 &generator, std::vector<float> &v) {
  float sum = 0.0f;
  for (size_t i = 0; i < v.size(); i++) {
    float val = (generator() % 256) / 100.0f;
    v[i] = val;
    sum += val * val;
  }
  return sum;
}

bool WeightDecayTest::operator()(size_t, cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto _context = pipeline->context();

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);

  std::vector<float> w1(_impl->w1_size);
  std::vector<float> w2(_impl->w2_size);
  std::vector<float> w3(_impl->w3_size);
  float res1 = fill_rand(generator, w1),  //
      res2 = fill_rand(generator, w2),    //
      res3 = fill_rand(generator, w3);
  float expected = (res1 + res2 + res3) * _impl->weight_decay_parameter;

  // mem alloc
  cnn_sr::CnnLayerGpuAllocationPool l1_alloc, l2_alloc, l3_alloc;

  /* clang-format off */
  l1_alloc.weights = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * _impl->w1_size);
  l2_alloc.weights = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * _impl->w2_size);
  l3_alloc.weights = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * _impl->w3_size);
  /* clang-format on */
  _context->write_buffer(l1_alloc.weights, (void *)&w1[0], true);
  _context->write_buffer(l2_alloc.weights, (void *)&w2[0], true);
  _context->write_buffer(l3_alloc.weights, (void *)&w3[0], true);

  // run
  float result = pipeline->weight_decay(l1_alloc, l2_alloc, l3_alloc,
                                        _impl->weight_decay_parameter);
  // std::cout << "expected: " << expected << "\tgot: " << result << std::endl;
  assert_equals(expected, result);
  return true;
}

//
//
}  // namespace specs
}  // namespace test
