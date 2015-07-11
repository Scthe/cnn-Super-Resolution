#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

#include <cstddef>  // for size_t

typedef struct _cl_event* cl_event;

namespace opencl {
class Kernel;
typedef size_t MemoryHandle;
class Context;

namespace utils {
struct ImageData;
}
}

// TODO move this to opencl::Context
const opencl::MemoryHandle gpu_nullptr = 1 << 30;

namespace cnn_sr {

struct LayerData;

struct CnnLayerGpuAllocationPool {
  // forward:
  opencl::MemoryHandle weights = gpu_nullptr;
  opencl::MemoryHandle bias = gpu_nullptr;
  opencl::MemoryHandle output = gpu_nullptr;
  // backpropagation:
  opencl::MemoryHandle deltas = gpu_nullptr;
  opencl::MemoryHandle grad_w = gpu_nullptr;
  opencl::MemoryHandle grad_b = gpu_nullptr;
  opencl::MemoryHandle previous_delta_w = gpu_nullptr;
  opencl::MemoryHandle previous_delta_b = gpu_nullptr;
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
  opencl::Context* context();

  cl_event extract_luma(opencl::utils::ImageData&, opencl::MemoryHandle&,
                        opencl::MemoryHandle&, bool, cl_event* ev = nullptr);

  cl_event execute_layer(opencl::Kernel&, const LayerData&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         opencl::MemoryHandle&, size_t, size_t,
                         cl_event* ev = nullptr);

  /**
   * @param  total_padding        difference in size between ground_truth image
   *                              and result. Should be equal to f1+f2+f3-3
   */
  cl_event mean_squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                              opencl::MemoryHandle gpu_buf_algo_res,
                              opencl::MemoryHandle& gpu_buf_target,
                              size_t ground_truth_w, size_t ground_truth_h,
                              size_t total_padding, cl_event* ev = nullptr);

  /** deltas for previous layer based on current layer */
  cl_event calculate_deltas(opencl::Kernel&, const LayerData&, const LayerData&,
                            cnn_sr::CnnLayerGpuAllocationPool&,
                            cnn_sr::CnnLayerGpuAllocationPool&,  //
                            size_t, size_t,                      //
                            cl_event* ev = nullptr);

  cl_event backpropagate(opencl::Kernel&, LayerData&,
                         opencl::MemoryHandle layer_input,
                         CnnLayerGpuAllocationPool&,  //
                         size_t layer_out_w, size_t layer_out_h,
                         cl_event* ev = nullptr);

  cl_event last_layer_delta(opencl::MemoryHandle gpu_buf_ground_truth,
                            opencl::MemoryHandle gpu_buf_algo_res,
                            opencl::MemoryHandle& gpu_buf_target,
                            float weight_decay,  //
                            size_t ground_truth_w, size_t ground_truth_h,
                            size_t total_padding, cl_event* ev = nullptr);

  float weight_decay(cnn_sr::CnnLayerGpuAllocationPool,
                     cnn_sr::CnnLayerGpuAllocationPool,
                     cnn_sr::CnnLayerGpuAllocationPool, float,
                     cl_event* ev = nullptr);

  cl_event update_parameters(LayerData&, CnnLayerGpuAllocationPool&,
                             float momentum, float learning_rate,
                             cl_event* ev = nullptr);

  ///
  /// misc. kernels
  ///
  cl_event subtract_mean(opencl::MemoryHandle, cl_event* ev = nullptr);
  float sum(opencl::MemoryHandle, bool squared = false, cl_event* ev = nullptr);
  cl_event subtract_from_all(opencl::MemoryHandle, float,
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
  bool allocation_has_right_size__(opencl::MemoryHandle, size_t,  //
                                   size_t, const char*);

 private:
  void pre_execute_layer_validation(const LayerData&, opencl::MemoryHandle,
                                    size_t, size_t);
  size_t element_count(opencl::MemoryHandle, size_t el_size);

 protected:
  opencl::Context* const _context;
  bool _initialized;

  /** Single 64bit number. Quite useful. */
  opencl::MemoryHandle _tmp_gpu_float = gpu_nullptr;

  opencl::Kernel* _luma_kernel_norm = nullptr;
  opencl::Kernel* _luma_kernel_raw = nullptr;
  opencl::Kernel* _mse_kernel = nullptr;
  opencl::Kernel* _sum_kernel = nullptr;
  opencl::Kernel* _sum_squared_kernel = nullptr;
  opencl::Kernel* _subtract_from_all_kernel = nullptr;
  opencl::Kernel* _last_layer_delta_kernel = nullptr;
  opencl::Kernel* _update_parameters_kernel = nullptr;
};
}

#endif /* DATA_PIPELINE_H   */
