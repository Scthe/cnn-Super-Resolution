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
 *  weights              := VECTOR[FLOAT], min size: f_spatial_size^2 * n_prev_filter_cnt * current_filter_count
 *  												 Each column for different filter(from 1 to current_filter_count)
 *  												 Each row for different point in range 0..f_spatial_size^2
 *  												 Each paragraph is 1 row of points  (f_spatial_size points)
 *  bias                 := VECTOR[FLOAT], min size: current_filter_count
 *
 * calcutated values:
 * 	out_w := input_w - f_spatial_size + 1
 * 	out_h := input_h - f_spatial_size + 1
 */
struct LayerData {
  /* clang-format on */

  LayerData(size_t n_prev_filter_cnt, size_t current_filter_count,
            size_t f_spatial_size, float* weights = nullptr,
            float* bias = nullptr);

  // LayerData(const LayerData&, float* weights, float* bias);

  static LayerData from_N_distribution(size_t n_prev_filter_cnt,
                                       size_t current_filter_count,
                                       size_t f_spatial_size,  //
                                       float mean_w = 0.0f, float sd_w = 0.001f,
                                       float mean_b = 0.0f, float sd_b = 0.0f);

  inline void get_output_dimensions(size_t* dim_arr, size_t input_w,
                                    size_t input_h) const {
    dim_arr[0] = input_w - f_spatial_size + 1;
    dim_arr[1] = input_h - f_spatial_size + 1;
  }
  inline const float* weights_ptr() const { return &weights[0]; }
  inline const float* bias_ptr() const { return &bias[0]; }

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
