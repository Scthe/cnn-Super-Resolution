#include "ConfigBasedDataPipeline.hpp"

#include <stdexcept>  // std::runtime_error
#include <cstdio>     // snprintf
#include <memory>     // std::unique_ptr

#include "LayerData.hpp"
#include "opencl/Context.hpp"
#include "opencl/UtilsOpenCL.hpp"
#include "Config.hpp"

namespace cnn_sr {

ConfigBasedDataPipeline::ConfigBasedDataPipeline(Config *cfg,
                                                 opencl::Context *context)
    : DataPipeline(context),
      _config(cfg),
      layer_data_1(1, cfg->n1, cfg->f1),
      layer_data_2(cfg->n1, cfg->n2, cfg->f2),
      layer_data_3(cfg->n2, 1, cfg->f3),
      _layer_1_kernel(nullptr),
      _layer_2_kernel(nullptr),
      _layer_3_kernel(nullptr),
      _layer_1_backpropagate_kernel(nullptr),
      _layer_2_backpropagate_kernel(nullptr),
      _layer_3_backpropagate_kernel(nullptr),
      _layer_1_deltas_kernel(nullptr),
      _layer_2_deltas_kernel(nullptr),
      _layer_3_deltas_kernel(nullptr) {}

void ConfigBasedDataPipeline::init(int load_flags) {
  if (!_config) {
    throw std::runtime_error(
        "Created ConfigBasedDataPipeline with incorrect(NULL) config.");
  }

  // TODO init weights/bias either file or random
  // LayerData::randomize_parameters(layer_data1, distr)

  load_kernels(load_flags);
  _initialized = true;
}

void ConfigBasedDataPipeline::load_kernels(int load_flags) {
  // call super
  DataPipeline::load_kernels(load_flags);

  bool load_layers = (load_flags & DataPipeline::LOAD_KERNEL_LAYERS) != 0,
       load_backp = (load_flags & DataPipeline::LOAD_KERNEL_BACKPROPAGATE) != 0;

  if (load_layers) {
    /* clang-format off */
    if (!_layer_1_kernel) _layer_1_kernel = create_layer_kernel(layer_data_1);
    if (!_layer_2_kernel) _layer_2_kernel = create_layer_kernel(layer_data_2);
    if (!_layer_3_kernel) _layer_3_kernel = create_layer_kernel(layer_data_3, 255);
    /* clang-format on */
  }

  if (load_backp) {
    /* clang-format off */
    if (!_layer_1_deltas_kernel) _layer_1_deltas_kernel = create_deltas_kernel(layer_data_1);
    if (!_layer_2_deltas_kernel) _layer_2_deltas_kernel = create_deltas_kernel(layer_data_2);
    if (!_layer_3_deltas_kernel) _layer_3_deltas_kernel = create_deltas_kernel(layer_data_3);
    /* clang-format on */

    /* clang-format off */
    if (!_layer_1_backpropagate_kernel)
      _layer_1_backpropagate_kernel = create_backpropagation_kernel(layer_data_1);
    if (!_layer_2_backpropagate_kernel)
      _layer_2_backpropagate_kernel = create_backpropagation_kernel(layer_data_2);
    if (!_layer_3_backpropagate_kernel)
      _layer_3_backpropagate_kernel = create_backpropagation_kernel(layer_data_3);
    /* clang-format on */
  }
}

cl_event ConfigBasedDataPipeline::forward(
    cnn_sr::CnnLayerGpuAllocationPool &layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_3_alloc,
    opencl::MemoryHandler *input, size_t input_w, size_t input_h,
    bool subtract_input_mean, cl_event *ev_to_wait_for) {
  //
  check_initialized(DataPipeline::LOAD_KERNEL_LAYERS);
  size_t l1_output_dim[2], l2_output_dim[2];
  layer_data_1.get_output_dimensions(l1_output_dim, input_w, input_h);
  layer_data_2.get_output_dimensions(l2_output_dim, l1_output_dim[0],
                                     l1_output_dim[1]);

  _context->block();

  cl_event ev;
  if (subtract_input_mean) {
    std::cout << "### Subtracting mean from input" << std::endl;
    ev = this->subtract_mean(input, ev_to_wait_for);
    ev_to_wait_for = &ev;
  }

  _context->block();

  // layer 1
  std::cout << "### Executing layer 1" << std::endl;
  cl_event finish_token1 =
      execute_layer(*_layer_1_kernel, layer_data_1, layer_1_alloc,  // layer cfg
                    input, input_w, input_h,                        // input
                    ev_to_wait_for);
  _context->block();

  // layer 2
  std::cout << "### Executing layer 2" << std::endl;
  cl_event finish_token2 = execute_layer(
      *_layer_2_kernel, layer_data_2, layer_2_alloc,             // layer cfg
      layer_1_alloc.output, l1_output_dim[0], l1_output_dim[1],  // input
      &finish_token1);
  _context->block();

  // layer 3
  std::cout << "### Executing layer 3" << std::endl;
  cl_event finish_token3 = execute_layer(
      *_layer_3_kernel, layer_data_3, layer_3_alloc,             // layer cfg
      layer_2_alloc.output, l2_output_dim[0], l2_output_dim[1],  // input
      &finish_token2);
  _context->block();

  return finish_token3;
}

cl_event ConfigBasedDataPipeline::mean_squared_error(
    opencl::MemoryHandler *gpu_buf_ground_truth,
    opencl::MemoryHandler *gpu_buf_algo_res,
    opencl::MemoryHandler *&gpu_buf_target,  //
    size_t ground_truth_w, size_t ground_truth_h, cl_event *ev_to_wait_for) {
  size_t padding = layer_data_1.f_spatial_size + layer_data_2.f_spatial_size +
                   layer_data_3.f_spatial_size - 3;
  return DataPipeline::mean_squared_error(
      gpu_buf_ground_truth, gpu_buf_algo_res, gpu_buf_target, ground_truth_w,
      ground_truth_h, padding, ev_to_wait_for);
}
}
