#include "TestSpecsDeclarations.hpp"

#include <random>  // for std::mt19937
#include <chrono>  // for random seed

#include "../../src/DataPipeline.hpp"

using namespace cnn_sr;

namespace test {
namespace specs {

///
/// PIMPL
///
struct LastLayerDeltaTestImpl {
  // const size_t algo_w = 1000, algo_h = 2000;
  const size_t algo_w = 6, algo_h = 6;
  const size_t padding = 4;
  const float weight_decay = 0.3f;
};

///
/// LastLayerDeltaTest
///

TEST_SPEC_PIMPL(LastLayerDeltaTest)

void LastLayerDeltaTest::init() {}

std::string LastLayerDeltaTest::name(size_t) { return "Last layer delta test"; }

size_t LastLayerDeltaTest::data_set_count() { return 1; }

bool LastLayerDeltaTest::operator()(size_t,
                                    cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto _context = pipeline->context();

  // total padding (from both sides) = padding*2
  const size_t total_padding = _impl->padding * 2,
               ground_truth_w = _impl->algo_w + total_padding,
               ground_truth_h = _impl->algo_h + total_padding,
               algo_size = _impl->algo_w * _impl->algo_h,
               ground_truth_size = ground_truth_w * ground_truth_h;

  std::vector<float> cpu_algo_res(algo_size);
  std::vector<float> cpu_expected(algo_size);
  std::vector<float> cpu_ground_truth(ground_truth_size);
  for (size_t i = 0; i < ground_truth_size; i++) {
    cpu_ground_truth[i] = 99999.0f;
  }

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed1);
  for (size_t i = 0; i < algo_size; i++) {
    size_t row = i / _impl->algo_w, col = i % _impl->algo_w,
           g_t_idx =
               (row + _impl->padding) * ground_truth_w + _impl->padding + col;
    float t = (generator() % 256) / 100.0f;
    // sigmoid etc
    float x = (generator() % 2560) / 1000.0f;
    float y = sigmoid(x);
    // fill expected buffer
    cpu_expected[i] = (y - t) * x * (1 - x) + _impl->weight_decay;
    cpu_ground_truth[g_t_idx] = t;
    cpu_algo_res[i] = y;
  }

  /* clang-format off */
  auto gpu_buf_ground_truth = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ground_truth_size);
  _context->write_buffer(gpu_buf_ground_truth, (void *)&cpu_ground_truth[0], true);
  auto gpu_buf_algo_res = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * algo_size);
  _context->write_buffer(gpu_buf_algo_res, (void *)&cpu_algo_res[0], true);
  /* clang-format on */
  opencl::MemoryHandle gpu_buf_out = gpu_nullptr;

  // exec
  pipeline->last_layer_delta(gpu_buf_ground_truth, gpu_buf_algo_res,
                             gpu_buf_out, _impl->weight_decay, ground_truth_w,
                             ground_truth_h, total_padding);
  assert_equals(pipeline, cpu_expected, gpu_buf_out);
  return true;
}

//
//
}  // namespace specs
}  // namespace test
