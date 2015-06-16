#include "LayerData.hpp"

#include <algorithm>  // for std::copy
#include <memory>     // for std::unique_ptr<float[]>
#include <chrono>     // for random seed
#include <cstdio>     // snprintf
// #include <exception>  // std::exception
#include <stdexcept>  // std::runtime_error
#include <ios>        // std::ios_base::failure
#include <iostream>

#include "json/gason.h"
#include "Utils.hpp"

namespace cnn_sr {

///
/// LayerData
///

LayerData::LayerData(size_t n_prev_filter_cnt, size_t current_filter_count,
                     size_t f_spatial_size, float* weights, float* bias)
    : n_prev_filter_cnt(n_prev_filter_cnt),
      current_filter_count(current_filter_count),
      f_spatial_size(f_spatial_size) {
  size_t weights_size = f_spatial_size * f_spatial_size * n_prev_filter_cnt *
                        current_filter_count;
  this->weights.reserve(weights_size);
  if (weights)
    std::copy(weights, weights + weights_size, back_inserter(this->weights));

  this->bias.reserve(current_filter_count);
  if (bias)
    std::copy(bias, bias + current_filter_count, back_inserter(this->bias));
}

LayerData LayerData::from_N_distribution(size_t n_prev_filter_cnt,
                                         size_t current_filter_count,
                                         size_t f_spatial_size,     //
                                         float mean_w, float sd_w,  //
                                         float mean_b, float sd_b) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> rand_generator_w(mean_w, sd_w);
  std::normal_distribution<float> rand_generator_b(mean_b, sd_b);

  size_t weights_size = f_spatial_size * f_spatial_size * n_prev_filter_cnt *
                        current_filter_count;
  std::unique_ptr<float[]> weights_buf(new float[weights_size]);
  for (size_t i = 0; i < weights_size; i++) {
    weights_buf[i] = rand_generator_w(generator);
  }

  std::unique_ptr<float[]> bias_buf(new float[current_filter_count]);
  for (size_t i = 0; i < current_filter_count; i++) {
    bias_buf[i] = rand_generator_b(generator);
  }

  return LayerData(n_prev_filter_cnt, current_filter_count, f_spatial_size,
                   weights_buf.get(), bias_buf.get());
}

void LayerData::validate(const LayerData& data) {
  size_t weights_size = data.f_spatial_size * data.f_spatial_size *
                        data.n_prev_filter_cnt * data.current_filter_count;
  if (data.weights.size() < weights_size) {
    char buf[255];
    snprintf(buf, 255,
             "Declared f_spatial_size(%d)*f_spatial_size(%d)"
             "*n_prev_filter_cnt(%d)*current_filter_count(%d)=%d"
             " is bigger then weights array (%d elements)."
             " Expected more elements in weights array",
             data.f_spatial_size, data.f_spatial_size, data.n_prev_filter_cnt,
             data.current_filter_count, weights_size, data.weights.size());
    throw std::runtime_error(buf);
  }

  if (data.bias.size() < data.current_filter_count) {
    char buf[255];
    snprintf(buf, 255,
             "Bias array(size=%d) should have equal size to "
             "current_filter_count(%d)",
             data.bias.size(), data.current_filter_count);
    throw std::runtime_error(buf);
  }
}

void LayerParametersIO::read(const char* const file,
                             std::vector<LayerData>& data,
                             const char* const* layer_keys,
                             size_t layer_key_count) {
  std::cout << "Loading layer parameters from: '" << file << "'" << std::endl;

  if (data.size() < layer_key_count) {
    throw std::runtime_error(
        "Allocate minimum as much LayerData objects as layer_keys You provide."
        " Note that LayerParametersIO::read normaly reads only"
        " values of weights and biases, rest of the data should already be"
        " provided from elsewhere (f.e. Config object)");
  }

  JsonValue value;
  JsonAllocator allocator;
  utils::read_json_file(file, value, allocator, JSON_OBJECT);

  LayerData* obj_ptr = &data[0];
  for (auto node : value) {
    auto key = node->key;
    // std::cout << key << std::endl;

    // check if this node represents one of allowed layer_keys
    bool key_ok = false;
    for (size_t i = 0; i < layer_key_count; i++) {
      key_ok |= strcmp(key, layer_keys[i]) == 0;
    }
    // std::cout << "found: " << key_ok << std::endl;
    if (!key_ok) continue;

    LayerData& obj = *obj_ptr;
    for (auto subnode : node->value) {
      JSON_READ_NUM_ARRAY(subnode, obj, weights)
      JSON_READ_NUM_ARRAY(subnode, obj, bias)
    }
    ++obj_ptr;
  }
}

//
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::LayerData& data) {
  os << "Layer:"
     << " n_prev_filter_cnt: " << data.n_prev_filter_cnt
     << ", current_filter_count: " << data.current_filter_count
     << ", f_spatial_size: " << data.f_spatial_size
     << ", weighs.size: " << data.weights.size()
     << ", bias.size: " << data.bias.size() << std::endl;
  return os;
}
