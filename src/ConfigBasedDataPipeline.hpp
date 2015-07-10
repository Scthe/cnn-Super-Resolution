#ifndef CONFIG_BASED_DATA_PIPELINE_H
#define CONFIG_BASED_DATA_PIPELINE_H

#include "DataPipeline.hpp"
#include "LayerData.hpp"

namespace cnn_sr {

struct Config;

class ConfigBasedDataPipeline : public DataPipeline {
 public:
  ConfigBasedDataPipeline(Config*, opencl::Context*);
  void init(int load_flags = DataPipeline::LOAD_KERNEL_ALL);

  cl_event forward(cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   cnn_sr::CnnLayerGpuAllocationPool&,  //
                   opencl::MemoryHandle, size_t, size_t, bool,
                   cl_event* ev = nullptr);

  cl_event mean_squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                              opencl::MemoryHandle gpu_buf_algo_res,
                              opencl::MemoryHandle& gpu_buf_target,
                              size_t ground_truth_w, size_t ground_truth_h,
                              cl_event* ev = nullptr);

 protected:
  void load_kernels(int load_flags);

 private:
  Config* const _config;
  LayerData layer_data_1;
  LayerData layer_data_2;
  LayerData layer_data_3;

  opencl::Kernel* _layer_1_kernel;
  opencl::Kernel* _layer_2_kernel;
  opencl::Kernel* _layer_3_kernel;
  opencl::Kernel* _layer_1_backpropagate_kernel;
  opencl::Kernel* _layer_2_backpropagate_kernel;
  opencl::Kernel* _layer_3_backpropagate_kernel;
  opencl::Kernel* _layer_1_deltas_kernel;
  opencl::Kernel* _layer_2_deltas_kernel;
  opencl::Kernel* _layer_3_deltas_kernel;
};
}

#endif /* CONFIG_BASED_DATA_PIPELINE_H   */
