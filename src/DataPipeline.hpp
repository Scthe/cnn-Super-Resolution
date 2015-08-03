#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

#include "pch.hpp"

// TODO move this to opencl::Context
const opencl::MemoryHandle gpu_nullptr = 1 << 30;

namespace cnn_sr {

struct LayerAllocationPool {
  /** Forward: weights, size: f*f*n*k */
  opencl::MemoryHandle weights = gpu_nullptr;
  /** Forward: bias, size: n */
  opencl::MemoryHandle bias = gpu_nullptr;

  /** Backpropagation: Accumulate gradients through out batch execution,
      size: f*f*n*k */
  opencl::MemoryHandle accumulating_grad_w = gpu_nullptr;
  /** Backpropagation: Accumulate gradients through out batch execution,
      size: f*f*n*k */
  opencl::MemoryHandle accumulating_grad_b = gpu_nullptr;
  /** Backpropagation-momentum: Deltas that we had after previous batch,
      size: f*f*n*k */
  opencl::MemoryHandle previous_batch_delta_w = gpu_nullptr;
  /** Backpropagation-momentum: Deltas that we had after previous batch,
      size: f*f*n*k */
  opencl::MemoryHandle previous_batch_delta_b = gpu_nullptr;
};

/**
 * Class used to execute various pipeline methods f.e.:
 * - luma extraction
 * - mean squared error
 * - all CNN methods
 *
 * This is quite low level - thin wrappers with validation mostly
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
  virtual void init(bool _optimize_for_small_data = false,
                    int load_flags = DataPipeline::LOAD_KERNEL_ALL);
  opencl::Context* context();

  /**
   * Take image, write it to GPU (gpu_buf_raw_img), and write luma channel
   * separately to gpu_buf_luma
   *
   * used buffers:
   * 	in  - NONE
   * 	out - param->gpu_buf_raw_img(with raw image data, all 3 channels)
   * 	      param->gpu_buf_luma(with luma channel of provided image)
   */
  cl_event extract_luma(opencl::utils::ImageData&, opencl::MemoryHandle&,
                        opencl::MemoryHandle&, bool, cl_event* ev = nullptr);

  /** Swap luma in image to specified set of values */
  cl_event swap_luma(opencl::utils::ImageData&,
                     opencl::MemoryHandle& gpu_buf_org_img,
                     opencl::MemoryHandle gpu_buf_new_luma,
                     opencl::MemoryHandle& target,  //
                     size_t new_luma_w, size_t new_luma_h,
                     cl_event* ev = nullptr);

  /**
   * Forward propagation for single layer.
   *
   * used buffers:
   * 	in  - layer.weights, layer.bias, this layer's input(that means previous
   *                                                             layer output)
   * 	out - layer.output
   */
  cl_event execute_layer(opencl::Kernel&, const LayerData&,
                         cnn_sr::LayerAllocationPool&, opencl::MemoryHandle&,
                         size_t, size_t, opencl::MemoryHandle&,
                         cl_event* ev = nullptr);

  /**
   * This function blocks.
   *
   * used buffers:
   * 	in  - orginal image luma, layer_3.output
   * 	out - this->_tmp_gpu_float
   *
   * @param  total_padding        difference in size between ground_truth image
   *                              and result. Should be equal to f1+f2+f3-3
   */
  cl_event squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                         size_t ground_truth_w, size_t ground_truth_h,
                         opencl::MemoryHandle gpu_buf_algo_res,
                         opencl::MemoryHandle& tmp_buffer, float& target,
                         size_t total_padding, cl_event* ev = nullptr);

  /**
   * Glorified sum to calculate weight decay. Important is that:
   * - this function blocks
   * - this functions applies weight_decay_parameter
   *
   * used buffers:
   * 	in  - layer_1.weights, layer_2.weights, layer_3.weights
   * 	out - this->_tmp_gpu_float
   */
  float weight_decay(cnn_sr::LayerAllocationPool, cnn_sr::LayerAllocationPool,
                     cnn_sr::LayerAllocationPool, float,
                     cl_event* ev = nullptr);

  /**
   * Deltas last layer
   *
   * used buffers:
   * 	in  - orginal image luma, layer_3.output
   * 	out - param->gpu_buf_target
   */
  cl_event last_layer_delta(opencl::MemoryHandle gpu_buf_ground_truth,
                            size_t ground_truth_w, size_t ground_truth_h,
                            opencl::MemoryHandle gpu_buf_algo_res,
                            opencl::MemoryHandle& gpu_buf_target,
                            float weight_decay,  //
                            size_t total_padding, cl_event* ev = nullptr);

  /**
   * Deltas for current layer based on next layer
   *
   * used buffers:
   * 	in  - next_layer.deltas, curr_layer.output, next_layer.weights
   * 	out - curr_layer.deltas
   */
  cl_event calculate_deltas(opencl::Kernel&,  //
                            const LayerData&, const LayerData&,
                            cnn_sr::LayerAllocationPool&,                 //
                            opencl::MemoryHandle&, opencl::MemoryHandle,  //
                            size_t, size_t,                               //
                            opencl::MemoryHandle,                         //
                            cl_event* ev = nullptr);

  /**
   * Calculate gradients of weights and bias
   *
   * used buffers:
   * 	in  - layer.deltas, this layer's input(that means previous layer output)
   * 	out - layer.grad_w, layer.grad_b
   */
  cl_event backpropagate(LayerData&, opencl::MemoryHandle layer_input,
                         opencl::MemoryHandle layer_deltas,
                         LayerAllocationPool&,  //
                         size_t layer_out_w, size_t layer_out_h,
                         cl_event* ev = nullptr);

  /**
   * Update weights and biases based on gradients and various factors like batch
   * size, momentum, learning rate. Note that we are both using
   * previous_delta_w/previous_delta_b to calculate his layers new
   * weights/biases(READ) and updating theirs values(WRITE).
   *
   * used buffers:
   * 	in  - layer.grad_w, layer.grad_b
   * 	out - layer.weights, layer.bias
   * 	in/out - layer.previous_delta_w, layer.previous_delta_b
   */
  cl_event update_parameters(LayerData&, LayerAllocationPool&,
                             size_t batch_size, float momentum,
                             float learning_rate, cl_event* ev = nullptr);

  ///
  /// misc. kernels
  ///

  /** Subtract mean value from all elements of the buffer. The mean parameter
   * will have the mean value */
  cl_event subtract_mean(opencl::MemoryHandle, float* mean = nullptr,
                         cl_event* ev = nullptr);

  /**
   * Sum all float in buffer. You may choose to square the values before adding
   * them up.
   */
  float sum(opencl::MemoryHandle, bool squared = false, cl_event* ev = nullptr);

  /** Subtract provided value from all elements of the buffer */
  cl_event subtract_from_all(opencl::MemoryHandle, float,
                             cl_event* ev = nullptr);

  ///
  /// kernel creation - ones that are not created during standard init
  ///
  /** @param  skip_relu:bool skip relu step, writing raw result */
  opencl::Kernel* create_layer_kernel(const LayerData&, bool);
  opencl::Kernel* create_deltas_kernel(const LayerData&);

  ///
  /// misc
  ///
  void print_buffer(opencl::MemoryHandle, const char* const, size_t);

 protected:
  void check_initialized(int kernel_load_flags);
  virtual void load_kernels(int load_flags);

  /** Either allocation has exact size or release it. Memory is deallocated
   * here, but we cannot allocate it with proper size since f.e. allocating
   * image is different then allocating normal buffer.
   */
  bool allocation_has_right_size__(opencl::MemoryHandle, size_t,  //
                                   size_t, const char*);

 private:
  void pre_execute_layer_validation(const LayerData&, opencl::MemoryHandle,
                                    size_t, size_t);
  size_t element_count(opencl::MemoryHandle, size_t el_size);

  /** General version. quite slow */
  cl_event execute_layer_full(opencl::Kernel&, const LayerData&,
                              cnn_sr::LayerAllocationPool&,
                              opencl::MemoryHandle&, size_t, size_t,
                              opencl::MemoryHandle&, cl_event* ev = nullptr);

  /** version optimized for case f_spatial_size==1 */
  cl_event execute_layer__f_eq_1(opencl::Kernel&, const LayerData&,
                                 cnn_sr::LayerAllocationPool&,
                                 opencl::MemoryHandle&, size_t, size_t,
                                 opencl::MemoryHandle&, cl_event* ev = nullptr);

  /** version optimized for case current_filter_count==1 */
  cl_event execute_output_layer(opencl::Kernel&, const LayerData&,
                                cnn_sr::LayerAllocationPool&,
                                opencl::MemoryHandle&, size_t, size_t,
                                opencl::MemoryHandle&, cl_event* ev = nullptr);

 protected:
  opencl::Context* const _context;
  bool _initialized;
  bool _optimize_for_small_data;

  /** Single float. Quite useful. */
  opencl::MemoryHandle _tmp_gpu_float = gpu_nullptr;

  opencl::Kernel* _luma_kernel_norm = nullptr;
  opencl::Kernel* _luma_kernel_raw = nullptr;
  opencl::Kernel* _swap_luma_kernel = nullptr;
  opencl::Kernel* _squared_error_kernel = nullptr;
  opencl::Kernel* _sum_kernel = nullptr;
  opencl::Kernel* _sum_squared_kernel = nullptr;
  opencl::Kernel* _subtract_from_all_kernel = nullptr;
  opencl::Kernel* _last_layer_delta_kernel = nullptr;
  opencl::Kernel* _update_parameters_kernel = nullptr;
  opencl::Kernel* _backpropagate_kernel = nullptr;
  opencl::Kernel* _backpropagate_kernel_opt = nullptr;
};
}

#endif /* DATA_PIPELINE_H   */
