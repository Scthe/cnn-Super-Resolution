#include "TestSpecsDeclarations.hpp"

// #include <vector>
// #include <random>  // for std::mt19937
// #include <chrono>  // for random seed

// #include "../src/opencl/Context.hpp"
#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"
#include "../../src/LayerData.hpp"
// #include "TestRunner.hpp"
// #include "TestDataProvider.hpp"
// #include "specs/TestSpecsDeclarations.hpp"
// #include "TestException.hpp"

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
  float output[INPUT_SIZE];
  float expected_derivative[INPUT_SIZE];
  for (size_t i = 0; i < INPUT_SIZE; i++) {
    float x = input_x[i];
    output[i] = sigmoid(x);
    expected_derivative[i] = x * (1 - x);
  }

// weights
// WEIGTHS_SIZE = f(l)*f(l)*n(l-1)*n(l)
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
      -0.631,	 0.301, -0.001,    -0.761, -0.021,  0.501
  };
/* clang-format on */

// deltas
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

  // gpu memory alloc
  cnn_sr::CnnLayerGpuAllocationPool prev_gpu_buf, curr_gpu_buf;
  /* clang-format off */
  curr_gpu_buf.deltas = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * DELTAS_SIZE);
  prev_gpu_buf.output = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * INPUT_SIZE);
  /* clang-format on */
  context->write_buffer(curr_gpu_buf.deltas, (void *)deltas, true);
  context->write_buffer(prev_gpu_buf.output, (void *)output, true);
  float bias[3] = {0.0f, 0.0f, 0.0f};
  curr_data.set_bias(bias);  // required to pass validation
  curr_data.set_weights(weights);

  // create kernel & run
  auto kernel = pipeline->create_deltas_kernel(curr_data);
  cl_event finish_token =
      pipeline->calculate_deltas(*kernel,                     //
                                 prev_data, curr_data,        //
                                 prev_gpu_buf, curr_gpu_buf,  //
                                 output_dim[0], output_dim[1]);

  std::cout << "kernel scheduled" << std::endl;
  context->block();
  std::cout << "kernel have finished " << prev_gpu_buf.deltas << std::endl;

  // read results
  float results[INPUT_SIZE];
  context->read_buffer(prev_gpu_buf.deltas, 0, sizeof(cl_float) * INPUT_SIZE,
                       (void *)results, true, &finish_token, 1);
  // context->read_buffer(prev_gpu_buf.deltas, 0, sizeof(cl_float) * 1,
  //  (void *)results, true, &finish_token, 1);
  for (size_t i = 0; i < INPUT_SIZE; i++) {
    float r = results[i];
    float expected = expected_derivative[i];
    std::cout << "[" << i << "] expected >\t" << expected << "\tvs " << r
              << std::endl;
  }

  return true;
}
}
}
