#include "TestSpecsDeclarations.hpp"

#include <random>  // for std::mt19937
#include <chrono>  // for random seed
#
#include "../../src/DataPipeline.hpp"
#include "../../src/LayerData.hpp"

namespace test {
namespace specs {

///
/// PIMPL
///
struct UpdateParametersTestImpl {
  const size_t n_prev_filter_cnt = 2, current_filter_count = 400,
               f_spatial_size = 5;
  // const size_t n_prev_filter_cnt = 2, current_filter_count = 2,
  //  f_spatial_size = 3;
  const float momentum = 0.8f, learning_rate = 0.001;

  void fill_data(std::mt19937& generator, std::vector<float>& data,
                 std::vector<float>& grad, std::vector<float>& previous_delta,
                 std::vector<float>& expected) {
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = (generator() % 2560) / 10.0f;
      grad[i] = (generator() % 2560) / 100.0f;
      previous_delta[i] = (generator() % 2560) / 10.0f;
      float delta = momentum * previous_delta[i] + learning_rate * grad[i];
      expected[i] = data[i] - delta;
    }
  }
};

///
/// UpdateParametersTest
///

TEST_SPEC_PIMPL(UpdateParametersTest)

void UpdateParametersTest::init() {}

size_t UpdateParametersTest::data_set_count() { return 1; }

std::string UpdateParametersTest::name(size_t) {
  return "Update parameters test";
}

bool UpdateParametersTest::operator()(size_t,
                                      cnn_sr::DataPipeline* const pipeline) {
  using namespace cnn_sr;
  assert_not_null(pipeline);
  auto context = pipeline->context();

  // create test data
  LayerData layer_data(_impl->n_prev_filter_cnt, _impl->current_filter_count,
                       _impl->f_spatial_size);
  size_t ws = layer_data.weight_size(), bs = layer_data.bias_size();
  std::vector<float> expected_w(ws),  //
      current_w(ws),                  //
      grad_w(ws),                     //
      previous_delta_w(ws);
  std::vector<float> expected_b(bs),  //
      current_b(bs),                  //
      grad_b(bs),                     //
      previous_delta_b(bs);
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);
  _impl->fill_data(generator, current_w, grad_w, previous_delta_w, expected_w);
  _impl->fill_data(generator, current_b, grad_b, previous_delta_b, expected_b);
  layer_data.set_weights(&current_w[0]);
  layer_data.set_bias(&current_b[0]);

  // mem allocs
  CnnLayerGpuAllocationPool gpu_alloc;
  /* clang-format off */
  gpu_alloc.weights          = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ws);
  gpu_alloc.grad_w           = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ws);
  gpu_alloc.previous_delta_w = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ws);
  gpu_alloc.bias             = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * bs);
  gpu_alloc.grad_b           = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * bs);
  gpu_alloc.previous_delta_b = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * bs);
  context->write_buffer(gpu_alloc.weights,          (void *)&current_w[0],        true);
  context->write_buffer(gpu_alloc.grad_w,           (void *)&grad_w[0],           true);
  context->write_buffer(gpu_alloc.previous_delta_w, (void *)&previous_delta_w[0], true);
  context->write_buffer(gpu_alloc.bias,             (void *)&current_b[0],        true);
  context->write_buffer(gpu_alloc.grad_b,           (void *)&grad_b[0],           true);
  context->write_buffer(gpu_alloc.previous_delta_b, (void *)&previous_delta_b[0], true);
  /* clang-format on */

  auto finish_token = pipeline->update_parameters(
      layer_data, gpu_alloc, _impl->momentum, _impl->learning_rate);

  // read results - weight
  context->read_buffer(gpu_alloc.weights, (void*)&grad_w[0], true,
                       &finish_token, 1);
  for (size_t i = 0; i < ws; i++) {
    float r = grad_w[i];
    float expected = expected_w[i];
    // std::cout << "[" << i << "] expected >\t" << expected << "\tgot> " << r
    // << std::endl;
    assert_equals(expected, r);
  }

  // read results - bias
  context->read_buffer(gpu_alloc.bias, (void*)&grad_b[0], true, &finish_token,
                       1);
  for (size_t i = 0; i < bs; i++) {
    float r = grad_b[i];
    float expected = expected_b[i];
    // std::cout << "[" << i << "] expected >\t" << expected << "\tgot> " << r
    // << std::endl;
    assert_equals(expected, r);
  }

  // TODO test if previous_delta is equal to current delta after running

  return true;
}

//
//
}  // namespace specs
}  // namespace test
