#include "ConfigBasedDataPipeline.hpp"

#include <random>     // for std::mt19937
#include <chrono>     // for random seed
#include <fstream>    // for parameters dump
#include <algorithm>  // f.e. std::minmax_element
#include "json/gason.h"

#include "Config.hpp"
#include "Utils.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

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
    this->epochs = load_parameters_file(_config->parameters_file);
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

  // layer 1
  if (print_steps) std::cout << "### Executing layer 1" << std::endl;
  cl_event finish_token1 =
      execute_layer(*_layer_1_kernel, layer_data_1, layer_1_alloc,  // layer cfg
                    input, input_w, input_h,                        // input
                    ev_to_wait_for);

  // layer 2
  if (print_steps) std::cout << "### Executing layer 2" << std::endl;
  cl_event finish_token2 = execute_layer(
      *_layer_2_kernel, layer_data_2, layer_2_alloc,             // layer cfg
      layer_1_alloc.output, l1_output_dim[0], l1_output_dim[1],  // input
      &finish_token1);

  // layer 3
  if (print_steps) std::cout << "### Executing layer 3" << std::endl;
  cl_event finish_token3 = execute_layer(
      *_layer_3_kernel, layer_data_3, layer_3_alloc,             // layer cfg
      layer_2_alloc.output, l2_output_dim[0], l2_output_dim[1],  // input
      &finish_token2);

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
  DataPipeline::backpropagate(layer_data_3,                            //
                              layer_2_alloc.output, layer_3_alloc,     //
                              layer_3_out_dim[0], layer_3_out_dim[1],  //
                              &event2_3);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 2nd layer"
              << std::endl;
  DataPipeline::backpropagate(layer_data_2,                            //
                              layer_1_alloc.output, layer_2_alloc,     //
                              layer_2_out_dim[0], layer_2_out_dim[1],  //
                              &event2_3);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 1st layer"
              << std::endl;
  auto event3_3 =
      DataPipeline::backpropagate(layer_data_1,                            //
                                  cnn_input, layer_1_alloc,                //
                                  layer_1_out_dim[0], layer_1_out_dim[1],  //
                                  &event2_3);
  _context->block();
  return event3_3;
}

void ConfigBasedDataPipeline::update_parameters(
    cnn_sr::CnnLayerGpuAllocationPool &layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool &layer_3_alloc, size_t batch_size,
    cl_event *ev_to_wait_for) {
  if (print_steps)
    std::cout << "### Updating weights and biases - 3rd layer" << std::endl;
  DataPipeline::update_parameters(layer_data_3, layer_3_alloc, batch_size,
                                  _config->momentum, _config->learning_rate[2],
                                  ev_to_wait_for);

  if (print_steps)
    std::cout << "### Updating weights and biases - 2nd layer" << std::endl;
  DataPipeline::update_parameters(layer_data_2, layer_2_alloc, batch_size,
                                  _config->momentum, _config->learning_rate[1],
                                  ev_to_wait_for);

  if (print_steps)
    std::cout << "### Updating weights and biases - 1st layer" << std::endl;
  DataPipeline::update_parameters(layer_data_1, layer_1_alloc, batch_size,
                                  _config->momentum, _config->learning_rate[0],
                                  ev_to_wait_for);

  _context->block();
  _context->zeros_float(layer_1_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_1_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_b, true);

  ++epochs;
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

size_t ConfigBasedDataPipeline::load_parameters_file(
    const char *const file_path) {
  size_t epochs = 0;
  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  utils::read_json_file(file_path, value, allocator, source, JSON_OBJECT);

  for (auto node : value) {
    auto key = node->key;
    // std::cout << key << std::endl;

    if (strcmp(key, "epoch") == 0 && node->value.getTag() == JSON_NUMBER) {
      epochs = (unsigned int)node->value.toNumber();
    } else if (strcmp(key, layer_parameters_key[0]) == 0) {
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
  return epochs;
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
    const char *const file_path,
    cnn_sr::CnnLayerGpuAllocationPool layer_1_alloc,
    cnn_sr::CnnLayerGpuAllocationPool layer_2_alloc,
    cnn_sr::CnnLayerGpuAllocationPool layer_3_alloc) {
  std::cout << "Saving parameters to: '" << file_path << "'" << std::endl;
  // read weights
  /* clang-format off */
  _context->read_buffer(layer_1_alloc.weights, (void *)&layer_data_1.weights[0], true);
  _context->read_buffer(layer_2_alloc.weights, (void *)&layer_data_2.weights[0], true);
  _context->read_buffer(layer_3_alloc.weights, (void *)&layer_data_3.weights[0], true);
  _context->read_buffer(layer_1_alloc.bias, (void *)&layer_data_1.bias[0], true);
  _context->read_buffer(layer_2_alloc.bias, (void *)&layer_data_2.bias[0], true);
  _context->read_buffer(layer_3_alloc.bias, (void *)&layer_data_3.bias[0], true);
  /* clang-format on */

  // write to file
  std::ofstream params_file;
  params_file.open(file_path);
  params_file << "{" << std::endl
              << "  \"epochs\": " << this->epochs << "," << std::endl
              << std::endl;

  /* clang-format off */
  dump_layer_parameters(params_file, layer_parameters_key[0], layer_data_1.weights, layer_data_1.bias);
  params_file << "," << std::endl;
  dump_layer_parameters(params_file, layer_parameters_key[1], layer_data_2.weights, layer_data_2.bias);
  params_file << "," << std::endl;
  dump_layer_parameters(params_file, layer_parameters_key[2], layer_data_3.weights, layer_data_3.bias);
  /* clang-format on */

  params_file << std::endl
              << "}";
}

///
/// Write image
///
void ConfigBasedDataPipeline::get_extreme_values(opencl::MemoryHandle buffer,
                                                 std::vector<float> &target,
                                                 float &min, float &max) {
  auto raw_memory = _context->raw_memory(buffer);
  size_t len = raw_memory->size / sizeof(cl_float);
  target.resize(len);
  _context->read_buffer(buffer, (void *)&target[0], true);

  auto min_max_it = std::minmax_element(target.cbegin(), target.cend());
  min = *min_max_it.first;
  max = *min_max_it.second;
}

void ConfigBasedDataPipeline::write_result_image(
    const char *const out_path,  //
    opencl::utils::ImageData &input_img,
    opencl::MemoryHandle input_img_3ch,  //
    opencl::MemoryHandle input_img_luma, float input_img_luma_mean,
    opencl::MemoryHandle new_luma, size_t luma_w, size_t luma_h) {
  /*
  // normally the result image will be gray for the most part.
  // what we have to do is to expand range of luma values
  // TODO or maybe just do more training ?
  std::vector<float> luma_values;
  float input_luma_min_val, input_luma_max_val,  //
      new_luma_min_val, new_luma_max_val;
  get_extreme_values(input_img_luma, luma_values,  //
                     input_luma_min_val, input_luma_max_val);
  get_extreme_values(new_luma, luma_values, new_luma_min_val, new_luma_max_val);
  // old luma has subtracted mean, reverse this
  float input_luma_mean = (input_luma_max_val - input_luma_min_val) / 2;
  input_luma_min_val += input_img_luma_mean;
  input_luma_max_val += input_img_luma_mean;
  std::cout << "old luma: " << input_luma_min_val  //
            << "\t- " << input_luma_max_val << std::endl;
  std::cout << "new luma: " << new_luma_min_val  //
            << "\t- " << new_luma_max_val << std::endl;
  // normalize luma values
  float norm_factor = new_luma_max_val - new_luma_min_val;
  for (float &f : luma_values) {
    f = (f - new_luma_min_val) / norm_factor;
    // f = 0.5f;
    // f = 1.0f;
  }
  auto new_luma_after_norm_extr =
      std::minmax_element(luma_values.cbegin(), luma_values.cend());
  std::cout << "new: " << *new_luma_after_norm_extr.first << "\t - "
            << *new_luma_after_norm_extr.second << std::endl;

  _context->block();
  _context->write_buffer(new_luma, &luma_values[0], true);
  _context->block();
  */

  // create result image
  opencl::MemoryHandle gpu_buf_target = gpu_nullptr;
  swap_luma(input_img, input_img_3ch, new_luma, gpu_buf_target, luma_w, luma_h);

  // read result
  size_t result_size = input_img.w * input_img.h * 3;  // 3 channels
  std::vector<unsigned char> result(result_size);
  _context->read_buffer(gpu_buf_target, (void *)&result[0], true);

  // write result
  opencl::utils::ImageData res_img(input_img.w, input_img.h, 3, &result[0]);
  opencl::utils::write_image(out_path, res_img);
}
}
