#include "ConfigBasedDataPipeline.hpp"

#include <random>   // for std::mt19937
#include <chrono>   // for random seed
#include <fstream>  // for parameters dump
#include "json/gason.h"

#include "Config.hpp"
#include "Utils.hpp"
#include "opencl/Context.hpp"

auto print_steps = false, print_info2 = false;

const char *const layer_parameters_key[3] = {"layer1", "layer2", "layer3"};

namespace cnn_sr {

ConfigBasedDataPipeline::ConfigBasedDataPipeline(Config &cfg,
                                                 opencl::Context *context)
    : DataPipeline(context),
      _config(&cfg),
      layer_data_1(1, cfg.n1, cfg.f1),
      layer_data_2(cfg.n1, cfg.n2, cfg.f2),
      layer_data_3(cfg.n2, 1, cfg.f3) {}

void ConfigBasedDataPipeline::init(int load_flags) {
  // init weights/bias
  if (_config->parameters_file && strlen(_config->parameters_file) > 0) {
    std::cout << "Loading layer parameters from: '" << _config->parameters_file
              << "'" << std::endl;
    load_parameters_file(_config->parameters_file);
  } else {
    std::cout
        << "No parameters file provided, initializing random weights and biases"
        << std::endl;
    fill_random_parameters(layer_data_1, _config->params_distr_1);
    fill_random_parameters(layer_data_2, _config->params_distr_2);
    fill_random_parameters(layer_data_3, _config->params_distr_3);
  }

  load_kernels(load_flags);
  _initialized = true;

  if (print_info2) std::cout << layer_data_1 << std::endl;
  if (print_info2) std::cout << layer_data_2 << std::endl;
  if (print_info2) std::cout << layer_data_3 << std::endl;
  LayerData::validate(layer_data_1);
  LayerData::validate(layer_data_2);
  LayerData::validate(layer_data_3);
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
    if (!_layer_3_kernel) _layer_3_kernel = create_layer_kernel(layer_data_3);
    /* clang-format on */
  }

  if (load_backp) {
    /* clang-format off */
    if (!_layer_1_deltas_kernel) _layer_1_deltas_kernel = create_deltas_kernel(layer_data_1);
    if (!_layer_2_deltas_kernel) _layer_2_deltas_kernel = create_deltas_kernel(layer_data_2);

    if (!_layer_1_backpropagate_kernel)
      _layer_1_backpropagate_kernel = create_backpropagation_kernel(layer_data_1);
    if (!_layer_2_backpropagate_kernel)
      _layer_2_backpropagate_kernel = create_backpropagation_kernel(layer_data_2);
    if (!_layer_3_backpropagate_kernel)
      _layer_3_backpropagate_kernel = create_backpropagation_kernel(layer_data_3);
    /* clang-format on */
  }
}

///
/// Pipeline: forward/backward
///
cl_event ConfigBasedDataPipeline::forward(
    cnn_sr::CnnLayerGpuAllocationPool &layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_3_alloc,
    opencl::MemoryHandle input, size_t input_w, size_t input_h,
    cl_event *ev_to_wait_for) {
  //
  check_initialized(DataPipeline::LOAD_KERNEL_LAYERS);
  size_t l1_output_dim[2], l2_output_dim[2];
  layer_data_1.get_output_dimensions(l1_output_dim, input_w, input_h);
  layer_data_2.get_output_dimensions(l2_output_dim, l1_output_dim[0],
                                     l1_output_dim[1]);
  _context->block();

  // layer 1
  if (print_steps) std::cout << "### Executing layer 1" << std::endl;
  cl_event finish_token1 =
      execute_layer(*_layer_1_kernel, layer_data_1, layer_1_alloc,  // layer cfg
                    input, input_w, input_h,                        // input
                    ev_to_wait_for);
  _context->block();

  // layer 2
  if (print_steps) std::cout << "### Executing layer 2" << std::endl;
  cl_event finish_token2 = execute_layer(
      *_layer_2_kernel, layer_data_2, layer_2_alloc,             // layer cfg
      layer_1_alloc.output, l1_output_dim[0], l1_output_dim[1],  // input
      &finish_token1);
  _context->block();

  // layer 3
  if (print_steps) std::cout << "### Executing layer 3" << std::endl;
  cl_event finish_token3 = execute_layer(
      *_layer_3_kernel, layer_data_3, layer_3_alloc,             // layer cfg
      layer_2_alloc.output, l2_output_dim[0], l2_output_dim[1],  // input
      &finish_token2);
  _context->block();

  return finish_token3;
}

cl_event ConfigBasedDataPipeline::backpropagate(
    cnn_sr::CnnLayerGpuAllocationPool &layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_3_alloc,
    opencl::MemoryHandle cnn_input, opencl::MemoryHandle gpu_buf_ground_truth,
    size_t ground_truth_w, size_t ground_truth_h,  //
    float weight_decay_value, cl_event *ev_to_wait_for) {
  // dimensions
  size_t layer_1_out_dim[2], layer_2_out_dim[2], layer_3_out_dim[2];
  layer_data_1.get_output_dimensions(layer_1_out_dim,  //
                                     ground_truth_w, ground_truth_h);
  layer_data_2.get_output_dimensions(layer_2_out_dim,  //
                                     layer_1_out_dim[0], layer_1_out_dim[1]);
  layer_data_3.get_output_dimensions(layer_3_out_dim,  //
                                     layer_2_out_dim[0], layer_2_out_dim[1]);

  // propagate deltas
  if (print_steps)
    std::cout << "### Calculating deltas for last layer" << std::endl;
  auto event2_1 =
      last_layer_delta(gpu_buf_ground_truth,  //
                       layer_3_alloc.output,  //
                       layer_3_alloc.deltas,  //
                       weight_decay_value,    //
                       ground_truth_w, ground_truth_h, ev_to_wait_for);

  if (print_steps)
    std::cout << "### Calculating deltas for 2nd layer" << std::endl;
  auto event2_2 = calculate_deltas(*_layer_2_deltas_kernel,                 //
                                   layer_data_2, layer_data_3,              //
                                   layer_2_alloc, layer_3_alloc,            //
                                   layer_3_out_dim[0], layer_3_out_dim[1],  //
                                   &event2_1);

  if (print_steps)
    std::cout << "### Calculating deltas for 1nd layer" << std::endl;
  auto event2_3 = calculate_deltas(*_layer_1_deltas_kernel,                 //
                                   layer_data_1, layer_data_2,              //
                                   layer_1_alloc, layer_2_alloc,            //
                                   layer_2_out_dim[0], layer_2_out_dim[1],  //
                                   &event2_2);

  _context->block();

  // gradient w, gradient b for all layers
  // TODO might as well run all in parallel
  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 3rd layer"
              << std::endl;
  auto event3_1 =
      DataPipeline::backpropagate2(layer_data_3,                            //
                                   layer_2_alloc.output, layer_3_alloc,     //
                                   layer_3_out_dim[0], layer_3_out_dim[1],  //
                                   &event2_3);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 2nd layer"
              << std::endl;
  auto event3_2 =
      DataPipeline::backpropagate2(layer_data_2,                            //
                                   layer_1_alloc.output, layer_2_alloc,     //
                                   layer_2_out_dim[0], layer_2_out_dim[1],  //
                                   &event3_1);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 1st layer"
              << std::endl;
  auto event3_3 =
      DataPipeline::backpropagate2(layer_data_1,                            //
                                   cnn_input, layer_1_alloc,                //
                                   layer_1_out_dim[0], layer_1_out_dim[1],  //
                                   &event3_2);
  return event3_3;
}

void ConfigBasedDataPipeline::update_parameters(
    cnn_sr::CnnLayerGpuAllocationPool &layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_3_alloc, size_t batch_size,
    cl_event *ev_to_wait_for) {
  // TODO might as well run all in parallel
  if (print_steps)
    std::cout << "### Updating weights and biases - 3rd layer" << std::endl;
  auto event1 = DataPipeline::update_parameters(
      layer_data_3, layer_3_alloc, batch_size, _config->momentum,
      _config->learning_rate[2], ev_to_wait_for);

  if (print_steps)
    std::cout << "### Updating weights and biases - 2nd layer" << std::endl;
  auto event2 = DataPipeline::update_parameters(
      layer_data_2, layer_2_alloc, batch_size, _config->momentum,
      _config->learning_rate[1], &event1);

  if (print_steps)
    std::cout << "### Updating weights and biases - 1st layer" << std::endl;
  DataPipeline::update_parameters(layer_data_1, layer_1_alloc, batch_size,
                                  _config->momentum, _config->learning_rate[0],
                                  &event2);

  _context->block();
  _context->zeros_float(layer_1_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_1_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_b, true);
}

float ConfigBasedDataPipeline::squared_error(
    opencl::MemoryHandle gpu_buf_ground_truth,
    opencl::MemoryHandle gpu_buf_algo_res,  //
    size_t ground_truth_w, size_t ground_truth_h, cl_event *ev_to_wait_for) {
  //
  size_t padding = layer_data_1.f_spatial_size + layer_data_2.f_spatial_size +
                   layer_data_3.f_spatial_size - 3;
  return DataPipeline::squared_error(gpu_buf_ground_truth, gpu_buf_algo_res,
                                     ground_truth_w, ground_truth_h, padding,
                                     ev_to_wait_for);
}

cl_event ConfigBasedDataPipeline::last_layer_delta(
    opencl::MemoryHandle gpu_buf_ground_truth,
    opencl::MemoryHandle gpu_buf_algo_res, opencl::MemoryHandle &gpu_buf_target,
    float weight_decay,  //
    size_t ground_truth_w, size_t ground_truth_h, cl_event *ev) {
  //
  size_t padding = _config->total_padding();
  return DataPipeline::last_layer_delta(
      gpu_buf_ground_truth, gpu_buf_algo_res, gpu_buf_target, weight_decay,
      ground_truth_w, ground_truth_h, padding, ev);
}

///
/// Parameters read
///
void ConfigBasedDataPipeline::fill_random_parameters(
    LayerData &data, ParametersDistribution &distr) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> rand_generator_w(distr.mean_w, distr.sd_w);
  std::normal_distribution<float> rand_generator_b(distr.mean_b, distr.sd_b);

  for (size_t i = 0; i < data.weight_size(); i++) {
    data.weights.push_back(rand_generator_w(generator));
  }
  for (size_t i = 0; i < data.bias_size(); i++) {
    data.bias.push_back(rand_generator_b(generator));
  }
}

void load_layer_parameters(JsonNode *node, LayerData &data) {
  for (auto subnode : node->value) {
    JSON_READ_NUM_ARRAY(subnode, data, weights);
    JSON_READ_NUM_ARRAY(subnode, data, bias);
  }
}

void ConfigBasedDataPipeline::load_parameters_file(
    const char *const file_path) {
  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  utils::read_json_file(file_path, value, allocator, source, JSON_OBJECT);

  for (auto node : value) {
    auto key = node->key;
    // std::cout << key << std::endl;

    if (strcmp(key, layer_parameters_key[0]) == 0) {
      load_layer_parameters(node, layer_data_1);
    } else if (strcmp(key, layer_parameters_key[1]) == 0) {
      load_layer_parameters(node, layer_data_2);
    } else if (strcmp(key, layer_parameters_key[2]) == 0) {
      load_layer_parameters(node, layer_data_3);
    } else {
      std::cout << "[Warning] Unknown key '" << key
                << "' in parameters file, only: '"    //
                << layer_parameters_key[0] << "', '"  //
                << layer_parameters_key[1] << "', '"  //
                << layer_parameters_key[2] << "' are allowed" << std::endl;
    }
  }
}

///
/// Parameters write
///
void dump_layer_parameters(std::ostream &os, const char *const key,
                           std::vector<float> &weights,
                           std::vector<float> &bias) {
  os << "  \"" << key << "\":{" << std::endl
     << "    \"weights\": [";
  cnn_sr::utils::dump_vector(os, weights);
  os << "]," << std::endl
     << "    \"bias\": [";
  cnn_sr::utils::dump_vector(os, bias);
  os << "]" << std::endl
     << "  }";
}

void ConfigBasedDataPipeline::write_params_to_file(
    const char *const file_path,    //
    std::vector<float> &weights_1,  //
    std::vector<float> &weights_2,  //
    std::vector<float> &weights_3,  //
    std::vector<float> &bias_1,     //
    std::vector<float> &bias_2,     //
    std::vector<float> &bias_3) {
  std::cout << "Saving parameters to: '" << file_path << "'" << std::endl;
  _context->block();

  // write to file
  std::ofstream params_file;
  params_file.open(file_path);
  params_file << "{" << std::endl;
  dump_layer_parameters(params_file, layer_parameters_key[0], weights_1,
                        bias_1);
  params_file << "," << std::endl;
  dump_layer_parameters(params_file, layer_parameters_key[1], weights_2,
                        bias_2);
  params_file << "," << std::endl;
  dump_layer_parameters(params_file, layer_parameters_key[2], weights_3,
                        bias_3);
  params_file << std::endl
              << "}";
}

/*
void ConfigBasedDataPipeline::dump_filters(
    const char *const path, size_t layer_id,
    cnn_sr::CnnLayerGpuAllocationPool gpu_alloc) {
  // read
  auto ws = data.weights_size();
  std::vector<float> w(ws);
  context.read_buffer(gpu_alloc.weights, (void *)&w[0], true);
  // dump
  char name_buf[255];
  std::vector<float> tmp(data.f_spatial_size * data.f_spatial_size);
  for (size_t n = 0; n < data.current_filter_count; n++) {
    for (size_t k = 0; k < data.n_prev_filter_cnt; k++) {
      size_t i = 0;
      for (size_t row = 0; row < data.f_spatial_size; row++) {
        for (size_t col = 0; col < data.f_spatial_size; col++) {
          size_t idx = ((row * data.f_spatial_size) + col) *
                       data.current_filter_count * data.n_prev_filter_cnt;
          idx += k * data.current_filter_count + n;
          tmp[i] = ws[idx];
          ++i;
        }
      }

      snprintf(name_buf, 255, "%s__%d_%d.png", path, k, n);
      dump_image(name_buf, data.f_spatial_size, tmp, true, 255.0f);
    }
  }
}
*/
}