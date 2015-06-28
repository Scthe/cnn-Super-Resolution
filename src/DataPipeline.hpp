#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

#include <cstddef>  // for size_t

typedef unsigned long long u64;
typedef struct _cl_event* cl_event;

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
struct Config;

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
  static int LOAD_KERNEL_LUMA;
  static int LOAD_KERNEL_LAYERS;
  static int LOAD_KERNEL_MISC;
  static int LOAD_KERNEL_NONE;
  static int LOAD_KERNEL_ALL;

  DataPipeline(Config*, opencl::Context*);
  void init(int load_flags = DataPipeline::LOAD_KERNEL_ALL);

  cl_event extract_luma(opencl::utils::ImageData&, opencl::MemoryHandler*&,
                        bool, cl_event* ev = nullptr);

  cl_event execute_cnn(LayerData&, LayerData&, LayerData&,
                       opencl::MemoryHandler*, opencl::MemoryHandler*&, size_t,
                       size_t, bool, cl_event* ev = nullptr);

  float mean_squared_error(opencl::MemoryHandler* gpu_buf_ground_truth,
                           opencl::MemoryHandler* gpu_buf_algo_res,
                           size_t ground_truth_w, size_t ground_truth_h,
                           cl_event* ev = nullptr);

  // misc. kernels
  cl_event subtract_mean(opencl::MemoryHandler*, cl_event* ev = nullptr);
  cl_event sum(opencl::MemoryHandler*, u64*, cl_event* ev = nullptr);
  cl_event subtract_from_all(opencl::MemoryHandler*, float,
                             cl_event* ev = nullptr);
  cl_event execute_layer(opencl::Kernel&, const LayerData&,
                         opencl::MemoryHandler*&, size_t, size_t,
                         opencl::MemoryHandler*&, cl_event* ev = nullptr);
  opencl::Kernel* create_layer_kernel(size_t current_filter_count,
                                      int result_multiply = 0);

 private:
  void check_initialized(int kernel_load_flags);
  void load_kernels(int load_flags);
  void pre_execute_layer_validation(const LayerData&, opencl::MemoryHandler*,
                                    size_t, size_t);

 private:
  Config* const _config;
  opencl::Context* const _context;
  bool _initialized;

  opencl::Kernel* _luma_kernel_norm;
  opencl::Kernel* _luma_kernel_raw;
  opencl::Kernel* _layer_1_kernel;
  opencl::Kernel* _layer_2_kernel;
  opencl::Kernel* _layer_3_kernel;
  opencl::Kernel* _sum_sq_kernel;
  opencl::Kernel* _sum_kernel;
  opencl::Kernel* _subtract_from_all_kernel;
};
}

#endif /* DATA_PIPELINE_H   */
