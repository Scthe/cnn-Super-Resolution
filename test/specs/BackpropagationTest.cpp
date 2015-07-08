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
struct BackpropagationTestImpl {
  DataPipeline *pipeline = nullptr;

// INPUT_SIZE = input_dim*n(l-1)
#define INPUT_SIZE 50
  float input[INPUT_SIZE] = {-0.083, -0.064,  //
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

// DELTAS_SIZE = output_dim * n(l)
#define DELTAS_SIZE 27
  float deltas[DELTAS_SIZE] = {0.122f, 0.083f, 0.064f,  // row 1, col 1
                               0.057f, 0.075f, 0.055f,  // row 1, col 2
                               0.025f, 0.058f, 0.138f,  // row 1, col 3

                               0.170f, 0.068f, 0.144f,  // row 2, col 1
                               0.121f, 0.013f, 0.176f,  // row 2, col 2
                               0.065f, 0.169f, 0.049f,  // row 2, col 3

                               0.003f, 0.181f, 0.051f,   // row 3, col 1
                               0.021f, 0.136f, 0.062f,   // row 3, col 2
                               0.066f, 0.165f, 0.176f};  // row 3, col 3
  float expected_bias[3] = {0.650f, 0.948f, 0.915f};
};

///
/// BackpropagationTest
///

TEST_SPEC_PIMPL(BackpropagationTest)

void BackpropagationTest::init(DataPipeline *pipeline) {
  _impl->pipeline = pipeline;
}

const char *BackpropagationTest::name() { return "Backpropagation test"; }

bool BackpropagationTest::operator()(opencl::Context *const context) {
#define WEIGHTS_SIZE 54

  auto pipeline = _impl->pipeline;

  // data for layer, needs filled up weights&bias to pass validation
  LayerData data(2, 3, 3);  // n_prev_filter_cnt/FILTER_CNT/f_spatial_size
  float w[WEIGHTS_SIZE], bias[10];
  data.set_bias(bias);
  data.set_weights(w);

  size_t output_dim[2] = {3, 3},
         input_dim[2] = {output_dim[0] + data.f_spatial_size - 1,
                         output_dim[1] + data.f_spatial_size - 1};

  // gpu memory alloc
  cnn_sr::CnnLayerGpuAllocationPool gpu_buf;
  opencl::MemoryHandler *gpu_buf_layer_input;
  /* clang-format off */
  gpu_buf.deltas = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * DELTAS_SIZE);
  gpu_buf_layer_input = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * INPUT_SIZE);
  /* clang-format on */
  context->write_buffer(gpu_buf.deltas, (void *)_impl->deltas, true);
  context->write_buffer(gpu_buf_layer_input, (void *)_impl->input, true);

  // create kernel & run
  auto kernel = pipeline->create_backpropagation_kernel(data);
  pipeline->backpropagate(*kernel, data, gpu_buf_layer_input, gpu_buf,  //
                          output_dim[0], output_dim[1]);

  std::cout << "[Info] kernel set to run, blocking" << std::endl;
  context->block();
  std::cout << "[Info] done" << std::endl;

  // for (size_t j = 0; j < data.weight_size(); j++) {}
  // for (size_t j = 0; j < 48 - 12; j++) {
  // std::cout << "w[" << j << "] " << data.grad_weights[j] << std::endl;
  // }

  for (size_t j = 0; j < data.bias_size(); j++) {
    float r = data.grad_bias[j];
    float expected = _impl->expected_bias[j];
    // std::cout << "b[" << j << "] expected >\t" << expected << "\tgot> " << r
    // << std::endl;
    assert_equals(expected, r);
  }

  return true;
}

//
//
}  // namespace specs
}  // namespace test
