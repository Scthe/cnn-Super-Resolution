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

struct CnnLayerGpuAllocationPool {
  opencl::MemoryHandler* weights = nullptr;
  opencl::MemoryHandler* bias = nullptr;
  opencl::MemoryHandler* output = nullptr;
  opencl::MemoryHandler* deltas = nullptr;
};

/**
 * Class used to execute various pipeline methods f.e.:
 * - luma extraction
 * - mean squared error
 * - whole cnn
 */
class DataPipeline {
 public:
  static int LOAD_KERNEL_LUMA;
  static int LOAD_KERNEL_LAYERS;
  static int LOAD_KERNEL_BACKPROPAGATE;
  static int LOAD_KERNEL_MISC;
  static int LOAD_KERNEL_NONE;
  static int LOAD_KERNEL_ALL;

  DataPipeline(opencl::Context*);
  virtual ~DataPipeline() {}
  virtual void init(int load_flags = DataPipeline::LOAD_KERNEL_ALL);

  cl_event extract_luma(opencl::utils::ImageData&, opencl::MemoryHandler*&,
                        opencl::MemoryHandler*&, bool, cl_event* ev = nullptr);

  cl_event execute_layer(opencl::Kernel&, const LayerData&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         opencl::MemoryHandler*&, size_t, size_t,
                         cl_event* ev = nullptr);

  /**
   * @param  total_padding        difference in size between ground_truth image
   *                              and result. Should be equal to f1+f2+f3-3
   */
  cl_event mean_squared_error(opencl::MemoryHandler* gpu_buf_ground_truth,
                              opencl::MemoryHandler* gpu_buf_algo_res,
                              opencl::MemoryHandler*& gpu_buf_target,
                              size_t ground_truth_w, size_t ground_truth_h,
                              size_t total_padding, cl_event* ev = nullptr);

  cl_event calculate_deltas(opencl::Kernel&, const LayerData&, const LayerData&,
                            cnn_sr::CnnLayerGpuAllocationPool&,
                            cnn_sr::CnnLayerGpuAllocationPool&,  //
                            size_t, size_t,                     //
                            cl_event* ev = nullptr);

  ///
  /// misc. kernels
  ///
  cl_event subtract_mean(opencl::MemoryHandler*, cl_event* ev = nullptr);
  cl_event sum(opencl::MemoryHandler*, u64*, cl_event* ev = nullptr);
  cl_event subtract_from_all(opencl::MemoryHandler*, float,
                             cl_event* ev = nullptr);

  ///
  /// kernel creation - ones that are not created during standard init
  ///
  opencl::Kernel* create_layer_kernel(const LayerData&,
                                      int result_multiply = 0);
  opencl::Kernel* create_deltas_kernel(const LayerData&);
  opencl::Kernel* create_backpropagation_kernel(const LayerData&);

 protected:
  void check_initialized(int kernel_load_flags);
  virtual void load_kernels(int load_flags);

  /** Either allocation has exact size or release it. Memory is deallocated
   * here, but we cannot allocate it with proper size since f.e. allocating
   * image is different then allocating normal buffer.
   * */
  bool allocation_has_right_size(opencl::MemoryHandler*, size_t);

 private:
  void pre_execute_layer_validation(const LayerData&, opencl::MemoryHandler*,
                                    size_t, size_t);

 protected:
  opencl::Context* const _context;
  bool _initialized;

  /** Single 64bit number. Quite useful. */
  opencl::MemoryHandler* _tmp_64bit = nullptr;

  opencl::Kernel* _luma_kernel_norm;
  opencl::Kernel* _luma_kernel_raw;
  opencl::Kernel* _mse_kernel;
  opencl::Kernel* _sum_kernel;
  opencl::Kernel* _subtract_from_all_kernel;
};
}

#endif /* DATA_PIPELINE_H   */
