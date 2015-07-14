#ifndef CONFIG_BASED_DATA_PIPELINE_H
#define CONFIG_BASED_DATA_PIPELINE_H

#include "DataPipeline.hpp"
#include "LayerData.hpp"

namespace cnn_sr {

struct ParametersDistribution;
struct Config;

class ConfigBasedDataPipeline : public DataPipeline {
 public:
  ConfigBasedDataPipeline(Config&, opencl::Context*);
  void init(int load_flags = DataPipeline::LOAD_KERNEL_ALL);

  cl_event forward(cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   opencl::MemoryHandle, size_t, size_t, bool,
                   cl_event* ev = nullptr);

  float squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                      opencl::MemoryHandle gpu_buf_algo_res,
                      size_t ground_truth_w, size_t ground_truth_h,
                      cl_event* ev = nullptr);

  /* clang-format off */
  /**
   * Steps:
   *   - calculate weight decay
   *   - calculate deltas for last layer
   *   - calculate deltas other layers in reverse order
   *   - backpropagate: calculate gradient w, gradient b for all layers
   *   - update weights and biases
   *
   * @param  layer_1_alloc        [description]
   * @param  layer_2_alloc        [description]
   * @param  layer_3_alloc        [description]
   * @param  cnn_input            input that was provided during forward step
   * @param  gpu_buf_ground_truth expected result
   * @param  ground_truth_w       width of both cnn_input and gpu_buf_ground_truth
   * @param  ground_truth_h       height of both cnn_input and gpu_buf_ground_truth
   * @param  ev_to_wait_for       [description]
   * @return                      [description]
   */
  cl_event backpropagate(cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         cnn_sr::CnnLayerGpuAllocationPool&,
                         opencl::MemoryHandle, opencl::MemoryHandle,
                         size_t, size_t, cl_event* ev_to_wait_for = nullptr);
  /* clang-format on */

 protected:
  void load_kernels(int load_flags);

 private:
  void fill_random_parameters(LayerData&, ParametersDistribution&);
  void load_parameters_file(const char* const);
  cl_event last_layer_delta(opencl::MemoryHandle, opencl::MemoryHandle,
                            opencl::MemoryHandle&, float, size_t, size_t,
                            cl_event* ev = nullptr);

 private:
  Config* const _config;
  LayerData layer_data_1;
  LayerData layer_data_2;
  LayerData layer_data_3;

  opencl::Kernel* _layer_1_kernel = nullptr;
  opencl::Kernel* _layer_2_kernel = nullptr;
  opencl::Kernel* _layer_3_kernel = nullptr;
  opencl::Kernel* _layer_1_backpropagate_kernel = nullptr;
  opencl::Kernel* _layer_2_backpropagate_kernel = nullptr;
  opencl::Kernel* _layer_3_backpropagate_kernel = nullptr;
  opencl::Kernel* _layer_1_deltas_kernel = nullptr;
  opencl::Kernel* _layer_2_deltas_kernel = nullptr;
};
}

#endif /* CONFIG_BASED_DATA_PIPELINE_H   */
