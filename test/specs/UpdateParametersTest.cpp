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
               f_spatial_size = 5, batch_size = 2;
  // const size_t n_prev_filter_cnt = 2, current_filter_count = 2,
  //  f_spatial_size = 3;
  const float momentum = 0.8f, learning_rate = 0.001;

  void create_data(std::mt19937 &generator, opencl::Context *context,
                   opencl::MemoryHandle &gpu_current_values,  //
                   opencl::MemoryHandle &gpu_grad,            //
                   opencl::MemoryHandle &gpu_previous_delta,  //
                   std::vector<float> &expected,
                   std::vector<float> &current_vals,
                   std::vector<float> &deltas) {
    size_t len = current_vals.size();
    std::vector<float> grad(len), previous_delta(len);
    for (size_t i = 0; i < len; i++) {
      current_vals[i] = (generator() % 2560) / 10.0f;
      grad[i] = (generator() % 2560) / 100.0f;
      previous_delta[i] = (generator() % 2560) / 10.0f;
      deltas[i] = momentum * previous_delta[i] + learning_rate * grad[i];
      expected[i] = current_vals[i] - (deltas[i] / batch_size);
    }

    // alloc
    /* clang-format off */
  gpu_current_values = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * len);
  gpu_grad           = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * len);
  gpu_previous_delta = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * len);
  context->write_buffer(gpu_current_values, (void *)&current_vals[0],   true);
  context->write_buffer(gpu_grad,           (void *)&grad[0],           true);
  context->write_buffer(gpu_previous_delta, (void *)&previous_delta[0], true);
    /* clang-format on */
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
                                      cnn_sr::DataPipeline *const pipeline) {
  using namespace cnn_sr;
  assert_not_null(pipeline);
  auto context = pipeline->context();

  // create test data
  LayerData layer_data(_impl->n_prev_filter_cnt, _impl->current_filter_count,
                       _impl->f_spatial_size);
  size_t ws = layer_data.weight_size(), bs = layer_data.bias_size();

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);

  LayerAllocationPool gpu_alloc;
  std::vector<float> expected_w(ws), current_w(ws), new_deltas_w(ws);
  std::vector<float> expected_b(bs), current_b(bs), new_deltas_b(bs);
  _impl->create_data(generator, context,
                     gpu_alloc.weights,           //
                     gpu_alloc.accumulating_grad_w,            //
                     gpu_alloc.previous_batch_delta_w,  //
                     expected_w, current_w, new_deltas_w);
  _impl->create_data(generator, context,
                     gpu_alloc.bias,              //
                     gpu_alloc.accumulating_grad_b,            //
                     gpu_alloc.previous_batch_delta_b,  //
                     expected_b, current_b, new_deltas_b);

  layer_data.set_weights(&current_w[0]);
  layer_data.set_bias(&current_b[0]);

  pipeline->update_parameters(layer_data, gpu_alloc, _impl->batch_size,
                              _impl->momentum, _impl->learning_rate);

  assert_equals(pipeline, expected_w, gpu_alloc.weights);
  assert_equals(pipeline, expected_b, gpu_alloc.bias);
  assert_equals(pipeline, new_deltas_w, gpu_alloc.previous_batch_delta_w);
  assert_equals(pipeline, new_deltas_b, gpu_alloc.previous_batch_delta_b);

  return true;
}

//
//
}  // namespace specs
}  // namespace test
