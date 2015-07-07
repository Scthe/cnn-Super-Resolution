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
  // cl_event backpropagate(opencl::Kernel&, const LayerData&,
                        //  opencl::MemoryHandler* layer_input,
                        //  CnnLayerGpuAllocationPool&, size_t layer_out_w,
                        //  size_t layer_out_h, cl_event* ev = nullptr);


  return true;
}

//
//
}  // namespace specs
}  // namespace test
