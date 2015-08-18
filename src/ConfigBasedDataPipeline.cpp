
#include "ConfigBasedDataPipeline.hpp"

#include <random>     // for std::mt19937
#include <chrono>     // for random seed
#include <fstream>    // for parameters dump
#include <cstring>    // for strcmp when reading json
#include <stdexcept>  // std::runtime_error
#include "json/gason.h"

#include "Config.hpp"
#include "pch.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

auto print_steps = false;

const char *const layer_parameters_key[3] = {"layer1", "layer2", "layer3"};

using namespace cnn_sr;

namespace cnn_sr {

ConfigBasedDataPipeline::ConfigBasedDataPipeline(Config &cfg,
                                                 opencl::Context *context)
    : DataPipeline(context),
      _config(&cfg),
      layer_data_1(1, cfg.n1, cfg.f1),
      layer_data_2(cfg.n1, cfg.n2, cfg.f2),
      layer_data_3(cfg.n2, 1, cfg.f3) {}

void ConfigBasedDataPipeline::init(int load_flags) {
  DataPipeline::init(load_flags);

  // init weights/bias
  if (!_config->parameters_file.empty()) {
    std::cout << "Loading layer parameters from: '" << _config->parameters_file
              << "'" << std::endl;
    this->epochs = load_parameters_file(_config->parameters_file.c_str());
    std::cout << "Previous epochs:  " << this->epochs << std::endl;
  } else {
    std::cout
        << "No parameters file provided, initializing random weights and biases"
        << std::endl;
    fill_random_parameters(layer_data_1, _config->params_distr_1);
    fill_random_parameters(layer_data_2, _config->params_distr_2);
    fill_random_parameters(layer_data_3, _config->params_distr_3);
  }
  LayerData::validate(layer_data_1);
  LayerData::validate(layer_data_2);
  LayerData::validate(layer_data_3);
}

void ConfigBasedDataPipeline::load_kernels(int load_flags) {
  DataPipeline::load_kernels(load_flags);

  bool load_layers = (load_flags & DataPipeline::LOAD_KERNEL_LAYERS) != 0,
       load_backp = (load_flags & DataPipeline::LOAD_KERNEL_BACKPROPAGATE) != 0;

  if (load_layers) {
    if (!_layer_1_kernel)
      _layer_1_kernel = create_layer_kernel(layer_data_1, false);
    if (!_layer_2_kernel)
      _layer_2_kernel = create_layer_kernel(layer_data_2, false);
    if (!_layer_3_kernel)
      _layer_3_kernel = create_layer_kernel(layer_data_3, true);
  }

  if (load_backp) {
    if (!_layer_1_deltas_kernel)
      _layer_1_deltas_kernel = create_deltas_kernel(layer_data_1);
    if (!_layer_2_deltas_kernel)
      _layer_2_deltas_kernel = create_deltas_kernel(layer_data_2);
  }
}

void ConfigBasedDataPipeline::set_mini_batch_size(size_t mini_batch_size) {
  _mini_batch_size = mini_batch_size;
  std::cout << "mini-batch size: " << _mini_batch_size << std::endl;
}

void ConfigBasedDataPipeline::allocate_buffers(size_t img_w, size_t img_h) {
  size_t l1_output_dim[2], l2_output_dim[2], l3_output_dim[2];
  layer_data_1.get_output_dimensions(l1_output_dim, img_w, img_h);
  layer_data_2.get_output_dimensions(l2_output_dim,  //
                                     l1_output_dim[0], l1_output_dim[1]);
  layer_data_3.get_output_dimensions(l3_output_dim,  //
                                     l2_output_dim[0], l2_output_dim[1]);

  size_t per_img0 = img_w * img_h,  //
      per_img1 = l1_output_dim[0] * l1_output_dim[1] *
                 layer_data_1.current_filter_count,
         per_img2 = l2_output_dim[0] * l2_output_dim[1] *
                    layer_data_2.current_filter_count,
         per_img3 = l3_output_dim[0] * l3_output_dim[1] *
                    layer_data_3.current_filter_count;

  /* clang-format off */
  _ground_truth_gpu_buf = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img0);
  _forward_gpu_buf = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img0);
  _out_1_gpu_buf   = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img1);
  _out_2_gpu_buf   = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img2);
  _out_3_gpu_buf   = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img3);
  _delta_1_gpu_buf = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img1);
  _delta_2_gpu_buf = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img2);
  _delta_3_gpu_buf = _context->allocate(CL_MEM_READ_WRITE, _mini_batch_size * 4 * per_img3);
  /* clang-format on */
}

///
/// Pipeline: forward/backward propagation wrappers
///

cl_event ConfigBasedDataPipeline::forward(LayerAllocationPool &layer_1_alloc,
                                          LayerAllocationPool &layer_2_alloc,
                                          LayerAllocationPool &layer_3_alloc,
                                          SampleAllocationPool &sample) {
  set_mini_batch_size(1);
  allocate_buffers(sample.input_w, sample.input_h);
  _context->copy_buffer(sample.input_luma, _forward_gpu_buf);
  // we use 0, since there is no offset
  return forward(layer_1_alloc,  //
                 layer_2_alloc,  //
                 layer_3_alloc,  //
                 sample.input_w, sample.input_h, 0);
}

float ConfigBasedDataPipeline::execute_batch(
    bool backpropagate__, GpuAllocationPool &gpu_alloc,
    std::vector<SampleAllocationPool *> &sample_set) {
  size_t i = 0;

  if (sample_set.empty() || _mini_batch_size == 0) {
    throw std::runtime_error("Batch cannot be empty");
  }
  size_t w = sample_set[0]->input_w, h = sample_set[0]->input_h;

  // allocate memory
  if (_out_1_gpu_buf == gpu_nullptr) {
    allocate_buffers(w, h);
  }

  float validation_error = 0.0f;  // only if executing validation set
  while (i < sample_set.size()) {
    // std::cout << "EXECUTING MINI-BATCH("
    // << (backpropagate__ ? "Backpropagate" : "Validation")
    // << "), start idx " << i << std::endl;

    // copy mini batch so that data is nicely aligned in memory
    size_t img_offset = 0, samples_in_batch = 0, j = i;
    while (samples_in_batch < _mini_batch_size && j < sample_set.size()) {
      SampleAllocationPool &sample = *sample_set[j];
      _context->copy_buffer(sample.input_luma, _forward_gpu_buf, img_offset,
                            nullptr, 0);
      _context->copy_buffer(sample.expected_luma, _ground_truth_gpu_buf,
                            img_offset, nullptr, 0);
      img_offset += sample.input_w * sample.input_h * 4;
      // img_offset += _context->raw_memory(sample.input_luma)->size;
      ++samples_in_batch;
      ++j;
    }

    // execute mini batch:
    for (size_t _k = 0; _k < _mini_batch_size && i < sample_set.size(); _k++) {
      SampleAllocationPool &sample = *sample_set[i];
      auto forward_ev = forward(gpu_alloc.layer_1,  //
                                gpu_alloc.layer_2,  //
                                gpu_alloc.layer_3,  //
                                sample.input_w, sample.input_h, _k);
      if (backpropagate__) {
        backpropagate(gpu_alloc.layer_1,                   //
                      gpu_alloc.layer_2,                   //
                      gpu_alloc.layer_3,                   //
                      sample.input_w, sample.input_h, _k,  //
                      &forward_ev);
        _context->block();
      } else {
        // we are executing validation set - schedule all squared_error calcs
        // (samples do not depend on each other, so we ignore event object)
        size_t padding = _config->total_padding();
        float validation_error__ = 0.0f;
        auto e = squared_error(_ground_truth_gpu_buf,               //
                               sample.input_w, sample.input_h, _k,  //
                               _out_3_gpu_buf, _tmp_gpu_float,
                               validation_error__, padding, &forward_ev);
        clWaitForEvents(1, &e);
        validation_error += validation_error__;
      }

      ++i;
    }

    // finish_mini_batch
    _context->block();
  }

  _context->block();

  return validation_error;
}

///
/// Pipeline: forward/backward propagation implementation
///
cl_event ConfigBasedDataPipeline::forward(
    LayerAllocationPool &layer_1_alloc,  //
    LayerAllocationPool &layer_2_alloc,  //
    LayerAllocationPool &layer_3_alloc,  //
    size_t sample_w, size_t sample_h, size_t sample_id) {
  //
  check_initialized(DataPipeline::LOAD_KERNEL_LAYERS);
  size_t l1_output_dim[2], l2_output_dim[2];
  layer_data_1.get_output_dimensions(l1_output_dim,  //
                                     sample_w, sample_h);
  layer_data_2.get_output_dimensions(l2_output_dim,  //
                                     l1_output_dim[0], l1_output_dim[1]);

  // if (sample_count > _mini_batch_size)
  // throw std::runtime_error("Allocation pool out of bounds exception");

  // layer 1
  if (print_steps) std::cout << "### Executing layer 1" << std::endl;
  cl_event finish_token1 =
      execute_layer(*_layer_1_kernel, layer_data_1, layer_1_alloc,  // layer cfg
                    _forward_gpu_buf, sample_w, sample_h, sample_id,  // input
                    _out_1_gpu_buf);

  // layer 2
  if (print_steps) std::cout << "### Executing layer 2" << std::endl;
  cl_event finish_token2 = execute_layer(
      *_layer_2_kernel, layer_data_2, layer_2_alloc,  // layer cfg
      _out_1_gpu_buf, l1_output_dim[0], l1_output_dim[1], sample_id,  // input
      _out_2_gpu_buf, &finish_token1);

  // layer 3
  if (print_steps) std::cout << "### Executing layer 3" << std::endl;
  cl_event finish_token3 = execute_layer(
      *_layer_3_kernel, layer_data_3, layer_3_alloc,  // layer cfg
      _out_2_gpu_buf, l2_output_dim[0], l2_output_dim[1], sample_id,  // input
      _out_3_gpu_buf, &finish_token2);

  return finish_token3;
}

cl_event ConfigBasedDataPipeline::backpropagate(
    cnn_sr::LayerAllocationPool &layer_1_alloc,
    cnn_sr::LayerAllocationPool &layer_2_alloc,
    cnn_sr::LayerAllocationPool &layer_3_alloc,          //
    size_t sample_w, size_t sample_h, size_t sample_id,  //
    cl_event *ev_to_wait_for) {
  // dimensions
  size_t layer_1_out_dim[2], layer_2_out_dim[2], layer_3_out_dim[2];
  layer_data_1.get_output_dimensions(layer_1_out_dim,  //
                                     sample_w, sample_h);
  layer_data_2.get_output_dimensions(layer_2_out_dim,  //
                                     layer_1_out_dim[0], layer_1_out_dim[1]);
  layer_data_3.get_output_dimensions(layer_3_out_dim,  //
                                     layer_2_out_dim[0], layer_2_out_dim[1]);

  // propagate deltas
  if (print_steps)
    std::cout << "### Calculating deltas for last layer" << std::endl;
  size_t padding = _config->total_padding();
  auto event2_1 = last_layer_delta(_ground_truth_gpu_buf,             //
                                   sample_w, sample_h, sample_id,     //
                                   _out_3_gpu_buf, _delta_3_gpu_buf,  //
                                   padding, ev_to_wait_for);

  if (print_steps)
    std::cout << "### Calculating deltas for 2nd layer" << std::endl;
  auto event2_2 = calculate_deltas(*_layer_2_deltas_kernel,     //
                                   layer_data_2, layer_data_3,  //
                                   layer_3_alloc,               //
                                   _delta_2_gpu_buf, _delta_3_gpu_buf,
                                   layer_3_out_dim[0], layer_3_out_dim[1],  //
                                   sample_id,                               //
                                   _out_2_gpu_buf, &event2_1);

  if (print_steps)
    std::cout << "### Calculating deltas for 1nd layer" << std::endl;
  auto event2_3 = calculate_deltas(*_layer_1_deltas_kernel,     //
                                   layer_data_1, layer_data_2,  //
                                   layer_2_alloc,               //
                                   _delta_1_gpu_buf, _delta_2_gpu_buf,
                                   layer_2_out_dim[0], layer_2_out_dim[1],  //
                                   sample_id,                               //
                                   _out_1_gpu_buf, &event2_2);

  // gradient w, gradient b for all layers
  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 3rd layer"
              << std::endl;
  auto event3_1 =
      DataPipeline::backpropagate(layer_data_3,  //
                                  _out_2_gpu_buf, _delta_3_gpu_buf,
                                  layer_3_alloc,                           //
                                  layer_3_out_dim[0], layer_3_out_dim[1],  //
                                  sample_id,                               //
                                  &event2_1);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 2nd layer"
              << std::endl;
  auto event3_2 =
      DataPipeline::backpropagate(layer_data_2,  //
                                  _out_1_gpu_buf, _delta_2_gpu_buf,
                                  layer_2_alloc,                           //
                                  layer_2_out_dim[0], layer_2_out_dim[1],  //
                                  sample_id,                               //
                                  &event2_2);

  if (print_steps)
    std::cout << "### Backpropagate(weights&bias gradients) - 1st layer"
              << std::endl;
  cl_event evs[3] = {event2_3, event3_1, event3_2};
  auto event3_3 =
      DataPipeline::backpropagate(layer_data_1,                            //
                                  _forward_gpu_buf, _delta_1_gpu_buf,      //
                                  layer_1_alloc,                           //
                                  layer_1_out_dim[0], layer_1_out_dim[1],  //
                                  sample_id,                               //
                                  evs, 3);

  return event3_3;
}

void ConfigBasedDataPipeline::update_parameters(
    cnn_sr::LayerAllocationPool &layer_1_alloc,
    cnn_sr::LayerAllocationPool &layer_2_alloc,
    cnn_sr::LayerAllocationPool &layer_3_alloc, size_t batch_size,
    cl_event *ev_to_wait_for) {
  if (print_steps)
    std::cout << "### Updating weights and biases - 3rd layer" << std::endl;
  DataPipeline::update_parameters(layer_data_3, layer_3_alloc, batch_size,
                                  _config->momentum,
                                  _config->weight_decay_parameter,
                                  _config->learning_rate[2], ev_to_wait_for);

  if (print_steps)
    std::cout << "### Updating weights and biases - 2nd layer" << std::endl;
  DataPipeline::update_parameters(layer_data_2, layer_2_alloc, batch_size,
                                  _config->momentum,
                                  _config->weight_decay_parameter,
                                  _config->learning_rate[1], ev_to_wait_for);

  if (print_steps)
    std::cout << "### Updating weights and biases - 1st layer" << std::endl;
  DataPipeline::update_parameters(layer_data_1, layer_1_alloc, batch_size,
                                  _config->momentum,
                                  _config->weight_decay_parameter,
                                  _config->learning_rate[0], ev_to_wait_for);

  // TODO optimize ?
  _context->block();
  _context->zeros_float(layer_1_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_w, true);
  _context->zeros_float(layer_1_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_2_alloc.accumulating_grad_b, true);
  _context->zeros_float(layer_3_alloc.accumulating_grad_b, true);

  ++epochs;
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
    utils::try_read_vector(*subnode, data.weights, "weights");
    utils::try_read_vector(*subnode, data.bias, "bias");
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

    if (utils::try_read_uint(*node, epochs, "epochs")) {
      continue;
    } else if (strcmp(key, layer_parameters_key[0]) == 0) {
      load_layer_parameters(node, layer_data_1);
    } else if (strcmp(key, layer_parameters_key[1]) == 0) {
      load_layer_parameters(node, layer_data_2);
    } else if (strcmp(key, layer_parameters_key[2]) == 0) {
      load_layer_parameters(node, layer_data_3);
    } else {
      std::cout << "[Warning] Unknown key '" << key << "' in parameters file"
                << std::endl;
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
    const char *const file_path,  //
    cnn_sr::LayerAllocationPool layer_1_alloc,
    cnn_sr::LayerAllocationPool layer_2_alloc,
    cnn_sr::LayerAllocationPool layer_3_alloc) {
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
/*
void ConfigBasedDataPipeline::create_lumas_delta_image(
    const char *const out_path, SampleAllocationPool &sample,
    AllocationItem &alloc) {
  opencl::MemoryHandle target = gpu_nullptr;
  size_t luma_w = sample.input_w - _config->total_padding(),
         luma_h = sample.input_h - _config->total_padding();
  // debug - last layer deltas
  // NOTE we do not have true expected luma, only one we started with
  size_t padding = _config->total_padding();
  auto event2_1 = DataPipeline::last_layer_delta(
      sample.input_luma, sample.input_w, sample.input_h,  //
      alloc.layer_3_output, target, padding);
  // read values
  std::vector<float> delta_values(luma_w * luma_h);
  _context->read_buffer(target, (void *)&delta_values[0], true, &event2_1);
  // write
  opencl::utils::write_image(out_path, &delta_values[0], luma_w, luma_h);
}
*/

void ConfigBasedDataPipeline::create_luma_image(const char *const out_path,
                                                opencl::MemoryHandle buffer,
                                                size_t luma_w, size_t luma_h) {
  std::vector<float> luma_data(luma_w * luma_h);
  _context->read_buffer(buffer, (void *)&luma_data[0], true);
  opencl::utils::write_image(out_path, &luma_data[0], luma_w, luma_h);
}

void ConfigBasedDataPipeline::write_result_image(
    const char *const out_path,  //
    opencl::utils::ImageData &input_img, SampleAllocationPool &sample) {
  std::cout << "Saving result image to: '" << out_path << "'" << std::endl;
  size_t luma_w = input_img.w - _config->total_padding(),
         luma_h = input_img.h - _config->total_padding();
  // create result image
  opencl::MemoryHandle gpu_buf_target = gpu_nullptr;
  swap_luma(input_img, sample.input_data, _out_3_gpu_buf, gpu_buf_target,
            luma_w, luma_h);

  // read result
  size_t result_size = input_img.w * input_img.h * 3;  // 3 channels
  std::vector<unsigned char> result(result_size);
  _context->read_buffer(gpu_buf_target, (void *)&result[0], true);

  // write result
  opencl::utils::ImageData res_img(input_img.w, input_img.h, 3, &result[0]);
  opencl::utils::write_image(out_path, res_img);

  // debug images
  /*
  std::cout << "[DEBUG] creating debug luma image" << std::endl;
  create_luma_image("data\\result_luma.png", alloc.layer_3_output, luma_w,
                    luma_h);
  std::cout << "[DEBUG] creating debug delta image (input vs output)"
            << std::endl;
  create_lumas_delta_image("data\\result_deltas.png", sample, alloc);
  */
}
}
