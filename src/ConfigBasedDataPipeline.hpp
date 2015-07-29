#ifndef CONFIG_BASED_DATA_PIPELINE_H
#define CONFIG_BASED_DATA_PIPELINE_H

#include "DataPipeline.hpp"
#include "LayerData.hpp"

namespace cnn_sr {

class ConfigBasedDataPipeline : public DataPipeline {
 public:
  ConfigBasedDataPipeline(Config&, opencl::Context*);
  void init(bool _optimize_for_small_data = false,
            int load_flags = DataPipeline::LOAD_KERNEL_ALL);
  inline const Config* config() { return _config; }

  cl_event forward(cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   opencl::MemoryHandle, size_t, size_t,
                   cl_event* ev = nullptr);

  float squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                      opencl::MemoryHandle gpu_buf_algo_res,
                      size_t ground_truth_w, size_t ground_truth_h,
                      cl_event* ev = nullptr);

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
  cl_event backpropagate(cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         opencl::MemoryHandle, opencl::MemoryHandle,
                         size_t, size_t,//
                         float, cl_event* ev_to_wait_for = nullptr);
  /* clang-format on */

  void write_params_to_file(const char* const file_path,  //
                            cnn_sr::CnnLayerGpuAllocationPool,
                            cnn_sr::CnnLayerGpuAllocationPool,
                            cnn_sr::CnnLayerGpuAllocationPool);

  /** update weights and biases*/
  void update_parameters(cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&, size_t batch_size,
                         cl_event* ev_to_wait_for = nullptr);

  void write_result_image(const char* const, opencl::utils::ImageData&,
                          opencl::MemoryHandle input_img_3ch,   //
                          opencl::MemoryHandle input_img_luma,  //
                          opencl::MemoryHandle new_luma, size_t, size_t);

  inline const LayerData* layer_1() { return &layer_data_1; }
  inline const LayerData* layer_2() { return &layer_data_2; }
  inline const LayerData* layer_3() { return &layer_data_3; }

 protected:
  void load_kernels(int load_flags);

 private:
  cl_event last_layer_delta(opencl::MemoryHandle, opencl::MemoryHandle,
                            opencl::MemoryHandle&, float, size_t, size_t,
                            cl_event* ev = nullptr);

  void fill_random_parameters(LayerData&, ParametersDistribution&);

  size_t load_parameters_file(const char* const);

  void create_luma_image(const char* const, opencl::MemoryHandle, size_t,
                         size_t);

  void create_lumas_delta_image(const char* const,
                                opencl::MemoryHandle expected_luma, size_t,
                                size_t,  //
                                opencl::MemoryHandle luma, size_t, size_t);

 private:
  Config* const _config;
  LayerData layer_data_1;
  LayerData layer_data_2;
  LayerData layer_data_3;
  size_t epochs = 0;

  opencl::Kernel* _layer_1_kernel = nullptr;
  opencl::Kernel* _layer_2_kernel = nullptr;
  opencl::Kernel* _layer_3_kernel = nullptr;
  opencl::Kernel* _layer_1_deltas_kernel = nullptr;
  opencl::Kernel* _layer_2_deltas_kernel = nullptr;
};
}

#endif /* CONFIG_BASED_DATA_PIPELINE_H   */
