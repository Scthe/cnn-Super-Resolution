#include "LayerData.hpp"

// #include <algorithm>  // for std::copy
#include <cstdio>     // snprintf
#include <stdexcept>  // std::runtime_error

namespace cnn_sr {

LayerData::LayerData(size_t n_prev_filter_cnt, size_t current_filter_count,
                     size_t f_spatial_size)
    : n_prev_filter_cnt(n_prev_filter_cnt),
      current_filter_count(current_filter_count),
      f_spatial_size(f_spatial_size) {
  // validation will pass if we set size to proper values, thus limiting it's
  // usefulness
  this->weights.reserve(this->weight_size());
  this->bias.reserve(this->bias_size());
}

void LayerData::validate(const LayerData& data) {
  if (data.weights.size() < data.weight_size()) {
    char buf[255];
    snprintf(buf, 255,
             "Declared f_spatial_size(%d)*f_spatial_size(%d)"
             "*n_prev_filter_cnt(%d)*current_filter_count(%d)=%d"
             " is bigger then weights array (%d elements)."
             " Expected more elements in weights array. ",
             data.f_spatial_size, data.f_spatial_size, data.n_prev_filter_cnt,
             data.current_filter_count, data.weight_size(),
             data.weights.size());
    throw std::runtime_error(buf);
  }

  if (data.bias.size() < data.bias_size()) {
    char buf[255];
    snprintf(buf, 255,
             "Bias array(size=%d) should have equal size to "
             "current_filter_count(%d).",
             data.bias.size(), data.bias_size());
    throw std::runtime_error(buf);
  }
}

///
/// get&set
///

void LayerData::set_weights(float* x) {
  if (x) std::copy(x, x + this->weight_size(), back_inserter(this->weights));
}

void LayerData::set_bias(float* x) {
  if (x) std::copy(x, x + this->bias_size(), back_inserter(this->bias));
}

void LayerData::get_output_dimensions(size_t* dim_arr, size_t input_w,
                                      size_t input_h) const {
  dim_arr[0] = input_w - f_spatial_size + 1;
  dim_arr[1] = input_h - f_spatial_size + 1;
}

size_t LayerData::weight_size() const {
  return f_spatial_size * f_spatial_size * n_prev_filter_cnt *
         current_filter_count;
}

size_t LayerData::bias_size() const { return current_filter_count; }

size_t LayerData::input_size(size_t input_w, size_t input_h) const {
  return input_w * input_h * n_prev_filter_cnt;
}

// namespace cnn_sr
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::LayerData& data) {
  os << "Layer {"
     << " previous filters: " << data.n_prev_filter_cnt
     << ", current filters: " << data.current_filter_count
     << ", f_spatial_size: " << data.f_spatial_size
     << ", weighs.size: " << data.weights.size()
     << ", bias.size: " << data.bias.size() << "}";
  return os;
}
