#include "TestSpecsDeclarations.hpp"

#include <random>  // for std::mt19937
#include <chrono>  // for random seed

#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"

namespace test {
namespace specs {

///
/// PIMPL
///
struct MeanSquaredErrorTestImpl {
  const size_t algo_w = 1000, algo_h = 2000;
  const size_t padding = 4;
};

///
/// MeanSquaredErrorTest
///

TEST_SPEC_PIMPL(MeanSquaredErrorTest)

void MeanSquaredErrorTest::init() {}

std::string MeanSquaredErrorTest::name(size_t) {
  return "Mean squared error test";
}

size_t MeanSquaredErrorTest::data_set_count() { return 1; }

bool MeanSquaredErrorTest::operator()(size_t,
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
    cpu_ground_truth[g_t_idx] = generator() % 256;
    cpu_algo_res[i] = (generator() % 2560) / 10.0f;
    // fill expected buffer
    double d = cpu_ground_truth[g_t_idx] - cpu_algo_res[i];
    cpu_expected[i] = d * d;
  }

  /* clang-format off */
  auto gpu_buf_ground_truth = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * ground_truth_size);
  _context->write_buffer(gpu_buf_ground_truth, (void *)&cpu_ground_truth[0], true);
  auto gpu_buf_algo_res = _context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * algo_size);
  _context->write_buffer(gpu_buf_algo_res, (void *)&cpu_algo_res[0], true);
  /* clang-format on */
  opencl::MemoryHandle gpu_buf_out = gpu_nullptr;

  // exec
  auto finish_token = pipeline->mean_squared_error(
      gpu_buf_ground_truth, gpu_buf_algo_res, gpu_buf_out, ground_truth_w,
      ground_truth_h, total_padding);

  // read & compare results
  std::vector<float> results(algo_size);
  _context->read_buffer(gpu_buf_out, (void *)&results[0], true, &finish_token,
                        1);
  for (size_t i = 0; i < algo_size; i++) {
    auto expected = cpu_expected[i];
    auto read_val = results[i];
    // std::cout << "expected: " << expected << "\tgot: " << read_val <<
    // std::endl;
    assert_equals(expected, read_val);
  }
  return true;
}

//
//
}  // namespace specs
}  // namespace test
