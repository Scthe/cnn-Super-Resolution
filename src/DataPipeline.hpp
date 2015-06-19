#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

#include "LayerExecutor.hpp"

namespace opencl {
class Kernel;
struct MemoryHandler;
class Context;

namespace utils {
struct ImageData;
}
}

namespace cnn_sr {
struct LayerData;

/**
 * Class used to execute various pipeline methods f.e.:
 * - luma extraction
 * - mean squared error
 * - whole cnn
 *
 * TODO manage gpu bufers here
 */
class DataPipeline {
 public:
  // typedef unsigned long long MseResult;
  typedef float MseResult;

  DataPipeline(Config*, opencl::Context*);
  void init();

  cl_event extract_luma(opencl::utils::ImageData&, opencl::MemoryHandler*&,
                        bool, cl_event* ev = nullptr);

  cl_event execute_cnn(LayerData&, LayerData&, LayerData&,
                       opencl::MemoryHandler*, opencl::MemoryHandler*&, size_t,
                       size_t, cl_event* ev = nullptr);

  MseResult mean_squared_error(opencl::MemoryHandler* gpu_buf_ground_truth,
                               opencl::MemoryHandler* gpu_buf_algo_res,
                               size_t ground_truth_w, size_t ground_truth_h,
                               cl_event* ev = nullptr);

 private:
  void check_initialized();

  bool _initialized;
  Config* const _config;
  opencl::Context* const _context;
  LayerExecutor _layer_executor;

  opencl::Kernel* _luma_kernel_norm;
  opencl::Kernel* _luma_kernel_raw;
  opencl::Kernel* _layer_1_kernel;
  opencl::Kernel* _layer_2_kernel;
  opencl::Kernel* _layer_3_kernel;
  opencl::Kernel* _sum_sq_kernel;
};
}

#endif /* DATA_PIPELINE_H   */
