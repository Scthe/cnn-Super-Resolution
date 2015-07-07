#ifndef LAYER_DATA_H
#define LAYER_DATA_H

#include <vector>
#include <cstddef>  // for size_t
#include <ostream>  // for operator<<

namespace cnn_sr {

/* clang-format off */
/**
 *
 *  Test data schema description (values for each layer provided after '/'):
 *
 *  n_prev_filter_cnt    := INT, filter count for previous layer, values: 1/n1/n2
 *  current_filter_count := INT, filter count for this layer, values: n1/n2/1
 *  f_spatial_size       := INT, spatial size, values: f1/f2/f3
 *  weights              := VECTOR[FLOAT], min size: weight_size
 *  												 Each column for different filter(from 1 to current_filter_count)
 *  												 Each row for different point in range 0..f_spatial_size^2
 *  												 Each paragraph is 1 row of points  (f_spatial_size points)
 *  bias                 := VECTOR[FLOAT], min size: bias_size
 *
 * calcutated values:
 *  input_size := input_w * input_h * n_prev_filter_cnt * current_filter_count
 *  out_w := input_w - f_spatial_size + 1
 *  out_h := input_h - f_spatial_size + 1
 *  weight_count := f_spatial_size^2 * n_prev_filter_cnt
 *  bias_count := current_filter_count
 */
struct LayerData {
  /* clang-format on */

  LayerData(size_t n_prev_filter_cnt, size_t current_filter_count,
            size_t f_spatial_size, float* weights = nullptr,
            float* bias = nullptr);

  static LayerData from_N_distribution(size_t n_prev_filter_cnt,
                                       size_t current_filter_count,
                                       size_t f_spatial_size,  //
                                       float mean_w = 0.0f, float sd_w = 0.001f,
                                       float mean_b = 0.0f, float sd_b = 0.0f);
  static void validate(const LayerData&);

  // get&set
  void set_weights(float*);
  void set_bias(float*);
  size_t input_size(size_t w, size_t h) const;
  void get_output_dimensions(size_t*, size_t w, size_t h) const;
  size_t weight_size() const;
  size_t bias_size() const;
  inline const float* weights_ptr() const { return &weights[0]; }
  inline const float* bias_ptr() const { return &bias[0]; }

 public:
  const size_t n_prev_filter_cnt;
  const size_t current_filter_count;
  const size_t f_spatial_size;
  std::vector<float> weights;
  std::vector<float> bias;
};

///
/// LayerParametersReader
///

/**
 * Only parameters for layers are weights and bias
 */
class LayerParametersIO {
 public:
  void read(const char* const, std::vector<LayerData>&,
            const char* const* layer_keys, size_t layer_key_count);
  // bool write(const char* const, LayerData**, size_t);
};
}

std::ostream& operator<<(std::ostream&, const cnn_sr::LayerData&);

#endif /* LAYER_DATA_H   */
