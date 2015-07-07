#include "TestSpecsDeclarations.hpp"

#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"
#include "../../src/LayerData.hpp"

using namespace cnn_sr;

namespace test {
namespace specs {

///
/// PIMPL
///
struct LayerDeltasTestImpl {
  DataPipeline *pipeline = nullptr;
};

///
/// NOTE: generating expected output is just checking if all weights, deltas
/// and outputs are read correctly. To this change following line:
///   'delta_for_filter[n] += delta * w * activation_func_derivative;'
/// to:
///   'delta_for_filter[n] += ONLY_ONE_OF_MULTIPLIERS'
/// compare results with:
///   weight: should be
///   activation_func_derivative: should be expected_derivative
///   delta: run LayerDeltasTest_script.py
/// if all of multipliers have correct value their produt will be ok.

// INPUT_SIZE = input_dim*n(l-1)
#define INPUT_SIZE 50
float input_x[INPUT_SIZE] = {-0.083, -0.064,  // pre sigmoid
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
   -0.220, -0.364,  0.711,   	0.281,  0.851, -1.001,
   -0.411,	 0.661, -0.831,    -0.091,  0.281, -0.341,

   -0.931,	 0.511,  0.141,    -0.591,  0.491, -0.921,
    0.291,	-0.211,  0.151,     0.491, -0.431, -0.321,
   -0.631,	 0.301, -0.001,    -0.761, -0.021,  0.501};
/* clang-format on */

// per filter n_prev_layer_filter: [-0.07799999999999996, -0.584]
// raw data:
// [-0.369,  0.025,  0.213, 0.236,  0.071, -0.429, 0.361, -0.055,  0.273, 0.229,
// 0.378, -0.178, -0.220, -0.364,  0.711, -0.411,	 0.661, -0.831,-0.931,
// 0.511,  0.141,0.291,	-0.211,  0.151,-0.631,	 0.301, -0.001]
// [0.058,  0.410, -0.068,-0.104,  0.161,  0.087,0.071,  0.431, -0.095,0.343,
// 0.114, -0.409,0.281,  0.851, -1.001,-0.091,  0.281, -0.341,-0.591,  0.491,
// -0.921,0.491, -0.431, -0.321,-0.761, -0.021,  0.501]

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

float expected_output[INPUT_SIZE] = {
    0.00263473,    -0.0025028,   // row 1
    -1.48462e-005, -0.00212133,  //
    -0.00452561,   -0.0102287,   //
    0.00128668,    -0.00821884,  //
    -0.000572977,  0.00198146,   //

    0.00246639,    0.00247348,   // row 2
    -0.00053351,   -0.00282118,  //
    -0.000531001,  -0.00501575,  //
    -0.0286956,    0.0129228,    //
    -0.00789852,   -0.002529,    //

    0.00110371,    -0.00214933,  // row 3
    0.000187628,   0.00399032,   //
    -0.0017117,    0.0054064,    //
    0.0289023,     -0.00391387,  //
    0.0199246,     -0.00979848,  //

    -0.0104161,    -0.0484205,    // row 4
    0.00600906,    -0.0254861,    //
    -0.119347,     -0.000239547,  //
    -0.000389471,  -0.0154712,    //
    -0.0118618,    0.0244768,     //

    0.010305,      -0.00323524,  // row 5
    -0.00330262,   -0.00287311,  //
    0.00824437,    -0.00538743,  //
    -0.00143925,   0.00453933,   //
    0.000386629,   -0.00124922,  //
};

///
/// LayerDeltasTest
///

TEST_SPEC_PIMPL(LayerDeltasTest)

void LayerDeltasTest::init(DataPipeline *pipeline) {
  _impl->pipeline = pipeline;
}

const char *LayerDeltasTest::name() { return "Layer deltas test"; }

bool LayerDeltasTest::operator()(opencl::Context *const context) {
#define IGNORED 10
  auto pipeline = _impl->pipeline;
  std::cout << "[DEBUG]" << std::endl;
  LayerData prev_data(IGNORED, 2, IGNORED);  // n(l-2), n(l-1), f(l-1)
  LayerData curr_data(2, 3, 3);              // n(l-1), n(l), f(l)

  // previous layer results - used to take care of sigmoid func.
  size_t output_dim[2] = {3, 3},
         input_dim[2] = {
             output_dim[0] + curr_data.f_spatial_size - 1,
             output_dim[1] + curr_data.f_spatial_size - 1,
         };

  // all variations with activation function
  float output[INPUT_SIZE];
  float expected_derivative[INPUT_SIZE];
  for (size_t i = 0; i < INPUT_SIZE; i++) {
    float x = input_x[i];
    output[i] = sigmoid(x);
    expected_derivative[i] = x * (1 - x);
  }

  // gpu memory alloc
  cnn_sr::CnnLayerGpuAllocationPool prev_gpu_buf, curr_gpu_buf;
  /* clang-format off */
  curr_gpu_buf.deltas = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * DELTAS_SIZE);
  prev_gpu_buf.output = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * INPUT_SIZE);
  /* clang-format on */
  context->write_buffer(curr_gpu_buf.deltas, (void *)deltas, true);
  context->write_buffer(prev_gpu_buf.output, (void *)output, true);
  float bias[3] = {0.0f, 0.0f, 0.0f};
  curr_data.set_bias(bias);  // required to pass layer validation
  curr_data.set_weights(weights);

  // create kernel & run
  auto kernel = pipeline->create_deltas_kernel(curr_data);
  cl_event finish_token =
      pipeline->calculate_deltas(*kernel,                     //
                                 prev_data, curr_data,        //
                                 prev_gpu_buf, curr_gpu_buf,  //
                                 output_dim[0], output_dim[1]);

  // read results
  float results[INPUT_SIZE];
  context->read_buffer(prev_gpu_buf.deltas, 0, sizeof(cl_float) * INPUT_SIZE,
                       (void *)results, true, &finish_token, 1);
  for (size_t i = 0; i < INPUT_SIZE; i++) {
    float r = results[i];
    float expected = expected_output[i];
    // std::cout << "[" << i << "] expected >\t" << expected << "\tgot> " << r
    // << std::endl;
    assert_equals(expected, r);
  }

  return true;
}

//
//
}  // namespace specs
}  // namespace test
