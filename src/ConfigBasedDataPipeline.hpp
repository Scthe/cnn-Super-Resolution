#ifndef CONFIG_BASED_DATA_PIPELINE_H
#define CONFIG_BASED_DATA_PIPELINE_H

#include "DataPipeline.hpp"
#include "LayerData.hpp"

namespace cnn_sr {

/**
 * All gpu buffer handles related to single image
 */
struct SampleAllocationPool {
  /** Raw 3 channel image loaded from hard drive */
  opencl::MemoryHandle input_data = gpu_nullptr;
  /** Single channel (luma) of size input_img_w*input_img_h */
  opencl::MemoryHandle input_luma = gpu_nullptr;
  /** Dimensions of original image*/
  size_t input_w, input_h;

  /** Training: Raw 3 channel image loaded from hard drive */
  opencl::MemoryHandle expected_data = gpu_nullptr;
  /** Training: luma to compare our result to */
  opencl::MemoryHandle expected_luma = gpu_nullptr;

  SampleAllocationPool() = default;

  // private:
  // SampleAllocationPool(const SampleAllocationPool&) = delete;
  // SampleAllocationPool& operator=(const SampleAllocationPool&) = delete;
};

/** Represents all general allocations that we will make */
struct GpuAllocationPool {
  LayerAllocationPool layer_1;
  LayerAllocationPool layer_2;
  LayerAllocationPool layer_3;

  std::vector<SampleAllocationPool> samples;
};

/**
 * Class that wraps all low level functions from DataPipeline into something
 * more usable
 */
class ConfigBasedDataPipeline : public DataPipeline {
 public:
  ConfigBasedDataPipeline(Config&, opencl::Context*);

  void init(int load_flags);

  void set_mini_batch_size(size_t);

  float execute_batch(bool backpropagate, GpuAllocationPool&,
                      std::vector<SampleAllocationPool*>&);

  cl_event forward(LayerAllocationPool& layer_1_alloc,  //
                   LayerAllocationPool& layer_2_alloc,  //
                   LayerAllocationPool& layer_3_alloc,  //
                   SampleAllocationPool& sample);

 private:
  void allocate_buffers(size_t, size_t);

  cl_event forward(LayerAllocationPool& layer_1_alloc,  //
                   LayerAllocationPool& layer_2_alloc,  //
                   LayerAllocationPool& layer_3_alloc,  //
                   size_t w, size_t h, size_t id);

  /* clang-format off */
  /**
   * General backpropagation steps:
   *   - calculate weight decay (NOTE: value expected as a paramter)
   *   - calculate deltas for last layer
   *   - calculate deltas other layers in reverse order
   *   - backpropagate: calculate gradient w, gradient b for all layers
   *   - update weights and biases (NOTE: requires explicit call to ConfigBasedDataPipeline::update_parameters(...))
   *
   * @param  layer_1_alloc        [description]
   * @param  layer_2_alloc        [description]
   * @param  layer_3_alloc        [description]
   * @param  cnn_input            input that was provided during forward step
   * @param  gpu_buf_ground_truth expected result
   * @param  ground_truth_w       width of both cnn_input and gpu_buf_ground_truth
   * @param  ground_truth_h       height of both cnn_input and gpu_buf_ground_truth
   * @param  weight_decay
   * @param  ev_to_wait_for       [description]
   * @return                      [description]
   */
  cl_event backpropagate(cnn_sr::LayerAllocationPool&,
                         cnn_sr::LayerAllocationPool&,
                         cnn_sr::LayerAllocationPool&,
                         size_t, size_t, size_t,
                         cl_event* ev_to_wait_for = nullptr);
  /* clang-format on */

 public:
  /** update weights and biases*/
  void update_parameters(cnn_sr::LayerAllocationPool&,
                         cnn_sr::LayerAllocationPool&,
                         cnn_sr::LayerAllocationPool&, size_t batch_size,
                         cl_event* ev_to_wait_for = nullptr);

  void write_params_to_file(const char* const file_path,  //
                            cnn_sr::LayerAllocationPool,
                            cnn_sr::LayerAllocationPool,
                            cnn_sr::LayerAllocationPool);

  void write_result_image(const char* const, opencl::utils::ImageData&,
                          SampleAllocationPool& sample);

  inline const Config* config() { return _config; }
  inline const LayerData* layer_1() { return &layer_data_1; }
  inline const LayerData* layer_2() { return &layer_data_2; }
  inline const LayerData* layer_3() { return &layer_data_3; }

 protected:
  void load_kernels(int load_flags);

 private:
  void fill_random_parameters(LayerData&, ParametersDistribution&);

  size_t load_parameters_file(const char* const);

  void create_luma_image(const char* const, opencl::MemoryHandle, size_t,
                         size_t);

  // void create_lumas_delta_image(const char* const, SampleAllocationPool& e,
  // AllocationItem&);

 private:
  Config* const _config;
  LayerData layer_data_1;
  LayerData layer_data_2;
  LayerData layer_data_3;
  size_t epochs = 0;
  size_t _mini_batch_size = 0;

  /* ground truth for batch */
  opencl::MemoryHandle _ground_truth_gpu_buf = gpu_nullptr;
  /** input for layer 1 */
  opencl::MemoryHandle _forward_gpu_buf = gpu_nullptr;
  /** outputs for layers */
  opencl::MemoryHandle _out_1_gpu_buf = gpu_nullptr,  //
      _out_2_gpu_buf = gpu_nullptr,                   //
      _out_3_gpu_buf = gpu_nullptr;
  /** deltas for layers */
  opencl::MemoryHandle _delta_1_gpu_buf = gpu_nullptr,  //
      _delta_2_gpu_buf = gpu_nullptr,                   //
      _delta_3_gpu_buf = gpu_nullptr;

  opencl::Kernel* _layer_1_kernel = nullptr;
  opencl::Kernel* _layer_2_kernel = nullptr;
  opencl::Kernel* _layer_3_kernel = nullptr;
  opencl::Kernel* _layer_1_deltas_kernel = nullptr;
  opencl::Kernel* _layer_2_deltas_kernel = nullptr;
};
}

#endif /* CONFIG_BASED_DATA_PIPELINE_H   */
