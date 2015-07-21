#include "TestSpecsDeclarations.hpp"

#include "../../src/DataPipeline.hpp"
#include "../../src/LayerData.hpp"

using namespace cnn_sr;

///
/// NOTE: generating expected output is just checking if all weights, deltas
/// and outputs are read correctly. To this change following line:
///   'delta_for_filter[n] += delta * w * activation_func_derivative;'
/// to:
///   'delta_for_filter[n] += ONLY_ONE_OF_MULTIPLIERS'
///
/// compare results with:
///   * weight: should be [-0.077999..., -0.584] (sum columns 1-3 for first
///             value, columns 4-6 for second)
///   * activation_func_derivative: should be expected_derivative
///                                 (code-generated)
///   * delta: run LayerDeltasTest_script.py
/// if all of multipliers have correct value their produt will be ok.
///

namespace test {
namespace specs {

///
/// PIMPL
///
struct LayerDeltasTestImpl {
// INPUT_SIZE = input_dim*n(l-1)
#define INPUT_SIZE 50
  float input_x[INPUT_SIZE] = {-0.083, -0.064,  //
                               0.075,  -0.055,  //
                               -0.058, -0.138,  //
                               -0.068, -0.144,  //
                               -0.013, 0.176,   //

                               0.169,  0.049,   //
                               0.181,  -0.051,  //
                               0.136,  -0.062,  //
                               -0.165, -0.176,  //
                               0.159,  -0.060,  //

                               -0.112, 0.228,   //
                               0.003,  -0.138,  //
                               -0.123, -0.027,  //
                               -0.102, -0.061,  //
                               0.242,  -0.069,  //

                               0.406,  0.419,   //
                               -0.442, 0.685,   //
                               -0.627, -0.489,  //
                               0.376,  0.563,   //
                               0.680,  -0.371,  //

                               0.121,  -0.075,   //
                               -0.103, 0.031,    //
                               0.106,  0.033,    //
                               -0.036, -0.052,   //
                               0.052,  -0.035};  //

// weights
// WEIGTHS_SIZE = f(l)*f(l)*n(l-1)*n(l)
// n(l)=3    |     n(l-1)=2
#define WEIGHTS_SIZE 54
  /* clang-format off */
float weights[WEIGHTS_SIZE] = {
   -0.369,  0.025,  0.213,     0.058,  0.410, -0.068,
    0.236,  0.071, -0.429,    -0.104,  0.161,  0.087,
    0.361, -0.055,  0.273,     0.071,  0.431, -0.095,

    0.229,  0.378, -0.178,     0.343,  0.114, -0.409,
   -0.220, -0.364,  0.711,     0.281,  0.851, -1.001,
   -0.411,   0.661, -0.831,    -0.091,  0.281, -0.341,

   -0.931,   0.511,  0.141,    -0.591,  0.491, -0.921,
    0.291,  -0.211,  0.151,     0.491, -0.431, -0.321,
   -0.631,   0.301, -0.001,    -0.761, -0.021,  0.501};
/* clang-format on */

// DELTAS_SIZE = output_dim * n(l)
#define DELTAS_SIZE 27
  float deltas[DELTAS_SIZE] = {0.122, 0.083, 0.064,   // row 1, col 1
                               0.057, 0.075, 0.055,   // row 1, col 2
                               0.025, 0.058, 0.138,   // row 1, col 3
                               0.170, 0.068, 0.144,   // row 2, col 1
                               0.121, 0.013, 0.176,   // row 2, col 2
                               0.065, 0.169, 0.049,   // row 2, col 3
                               0.003, 0.181, 0.051,   // row 3, col 1
                               0.021, 0.136, 0.062,   // row 3, col 2
                               0.066, 0.165, 0.176};  // row 3, col 3

  /* clang-format off */
  std::vector<float> expected_output = {
    0,                      0,
    -0.000213999,           0,
    0,                      0,
    0,                      0,
    0,                      0.013663,

    0.017562,               0.05308,
    -0.00359898,            0,
    -0.004519,              0,
    0,                      0,
    -0.059068,              0,

    0,                     -0.012211,
    0.06273,                0,
    0,                      0,
    0,                      0,
    0.108619,               0,

    -0.043191,             -0.198902,
    0,                     -0.118114,
    0,                      0,
    -0.00165999,           -0.062883,
    -0.054512,              0,

    0.096889,               0,
    0,                     -0.095646,
    0.086999,              -0.168827,
    0,                      0,
    0.007843,               0
  };
  /* clang-format on */
};

///
/// LayerDeltasTest
///

TEST_SPEC_PIMPL(LayerDeltasTest)

void LayerDeltasTest::init() {}

std::string LayerDeltasTest::name(size_t) { return "Layer deltas test"; }

size_t LayerDeltasTest::data_set_count() { return 1; }

bool LayerDeltasTest::operator()(size_t, cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto context = pipeline->context();

  const size_t IGNORED = 10;

  // data for layer, needs filled up weights&bias to pass validation
  LayerData prev_data(IGNORED, 2, IGNORED);  // n(l-2), n(l-1), f(l-1)
  LayerData curr_data(2, 3, 3);              // n(l-1), n(l), f(l)
  float bias[3] = {0.0f, 0.0f, 0.0f};
  curr_data.set_bias(bias);
  curr_data.set_weights(_impl->weights);

  // previous layer results - used to take care of sigmoid func.
  size_t output_dim[2] = {3, 3};

  // all variations with activation function
  float output[INPUT_SIZE];
  std::vector<float> expected_derivative(INPUT_SIZE);
  size_t derivative_repeat_cnt = curr_data.f_spatial_size *
                                 curr_data.f_spatial_size *
                                 curr_data.current_filter_count;
  for (size_t i = 0; i < INPUT_SIZE; i++) {
    float x = _impl->input_x[i];
    output[i] = activation_function(x);
    expected_derivative[i] =
        activation_function_derivative(x) * derivative_repeat_cnt;
  }

  // gpu memory alloc
  cnn_sr::CnnLayerGpuAllocationPool prev_gpu_buf, curr_gpu_buf;
  /* clang-format off */
  curr_gpu_buf.deltas = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * DELTAS_SIZE);
  prev_gpu_buf.output = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * INPUT_SIZE);
  /* clang-format on */
  context->write_buffer(curr_gpu_buf.deltas, (void *)_impl->deltas, true);
  context->write_buffer(prev_gpu_buf.output, (void *)output, true);

  // create kernel & run
  auto kernel = pipeline->create_deltas_kernel(prev_data);
  pipeline->calculate_deltas(*kernel,                     //
                             prev_data, curr_data,        //
                             prev_gpu_buf, curr_gpu_buf,  //
                             output_dim[0], output_dim[1]);
  assert_equals(pipeline, _impl->expected_output, prev_gpu_buf.deltas);

  // sub test with expected_derivative
  // assert_equals(pipeline, expected_derivative, prev_gpu_buf.deltas);

  return true;
}

//
//
}  // namespace specs
}  // namespace test
