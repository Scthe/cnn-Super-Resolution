
/* clang-format off */
/**
 *
 * In following notation (l), (l-1) describes relative layer and [...] lower indices.
 *
 * Algo for delta_ijn on layer (l-1), where
 *   i = 0..output_w(l-1),
 *   j = 0..output_h(l-1),
 *   n = 0..filter_count(l-1):
 *
 * delta[i,j,n](l-1) = 0
 * for a = 0..spatial_size(l+1):
 *   for b = 0..spatial_size(l+1):
 *     for k = 0..filter_count(l+1):
 *       delta[i,j,n](l-1) += \
 *         w[abnk](l-1)               # (1) weight of edge between [i,j,n](l-1) and [i+a,j+b,k](l)
 *         * delta[i+a,j+b,k](l)      # (2) error term for [i+a,j+b,k](l)
 *         * f`( x[i+a,j+b,n](l-1) )  # (3) derivative of activation function
 *         #* f`(x[i,j,n](l-1) )      # nope? - (3) derivative of activation function at measured point
 *
 * TODO in (3) should index be x[i+a,j+b,n] or x[i,j,n]?
 * TODO in popular notation it's [i-a,j-b,_], is this just because matlab's
 *      native conv2 expects transposition ?
 *
 * macros:
 * 	CURRENT_FILTER_COUNT                   filter_count(l-1)
 *
 * @param  float*      deltas_next_layer   size: output_w(l) * output_w(l) * filter_count(l)
 * @param  float*      layer_output        size: output_w(l-1) * output_w(l-1) * filter_count(l-1)
 * @param  float*      target              size: output_w(l-1) * output_w(l-1) * filter_count(l-1)
 * @param  float*      W                   weights between (l-1) and (l).
 *                                         WARN: w3 is between (l2) and (l3), w2 -> (l1) and (l2), w1 -> (input) and (l1)
 *                                         size: f_spatial_size*f_spatial_size*filter_count(l-1)*filter_count(l)
 * @param  uint        f_spatial_size      spatial/kernel size for (l-1)
 * @param  uint        f_next_spatial_size spatial/kernel size for (l)
 * @param  uint        n_next_filter_cnt   filter_count(l)
 * @param  uint        layer_out_w         output_w(l-1)
 * @param  uint        layer_out_h         output_h(l-1)
 * @return {[type]}                        [description]
 */
/* clang-format on */
__kernel void main(__read_only __global float* deltas_next_layer,  //
                   __read_only __global float* layer_output,       //
                   __global float* target,                         //
                   __read_only __global float* W,                  //
                   uint f_spatial_size,                            //
                   uint f_next_spatial_size,                       //
                   uint n_next_filter_cnt,                         //
                   uint layer_out_w, uint layer_out_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const int idx = ((pos.y * layer_out_w) + pos.x) * CURRENT_FILTER_COUNT;
  const int2 next_layer_out = {layer_out_w - f_next_spatial_size + 1,
                               layer_out_h - f_next_spatial_size + 1};
  const int half_f_next_spatial_size = (f_next_spatial_size - 1) / 2;

  // when working with source with at least 3 dimensions we have
  // to multiply by f_spatial_size for indexing purposes.
  // It's just 1 when working with 2D source.
  const int _2nd_dim_size = (CURRENT_FILTER_COUNT > 1) ? f_spatial_size : 1;

  // range check for i,j
  if (pos.x >= 0 && pos.x < layer_out_w &&  //
      pos.y >= 0 && pos.y < layer_out_h) {
    // zeroed result cache
    float vals_by_filter[CURRENT_FILTER_COUNT];
    for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
      vals_by_filter[n] = 0.0f;
    }

    for (size_t b = 0; b < f_next_spatial_size; b++) {
      for (size_t a = 0; a < f_next_spatial_size; a++) {
        int2 point_pos = {pos.x + a, pos.y + b};
        int base_point_idx =
            ((point_pos.y * layer_out_w) + point_pos.x) * CURRENT_FILTER_COUNT;
        int base_W_idx = ((b * f_spatial_size) + a) * _2nd_dim_size;

        for (size_t k = 0; k < n_next_filter_cnt; k++) {
          // (2) delta[i+a,j+b,k](l)
          // this requires us to map curent point_pos to next layer,
          // but some of the point may not be in range
          float delta = 0;
          int2 next_layer_pos = {point_pos.x - half_f_next_spatial_size,
                                 point_pos.y - half_f_next_spatial_size};
          if (next_layer_pos.x >= 0 && next_layer_pos.x < next_layer_out.x &&
              next_layer_pos.y >= 0 && next_layer_pos.y < next_layer_out.y) {
            int next_layer_idx =
                ((next_layer_pos.y * next_layer_out.x) + next_layer_pos.x) *
                    n_next_filter_cnt +
                k;
            delta = deltas_next_layer[next_layer_idx];
          }

          for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
            // (1) w[abnk](l-1)
            int w_idx = (base_W_idx + n) * CURRENT_FILTER_COUNT + k;
            float w = W[w_idx];

            // (3) f`( x[i+a,j+b,n](l-1) )
            // y_ijn := activation (value after sigmoid application)
            // x_ijn := value before sigmoid application
            float y_ijn = layer_output[base_point_idx + n];
            float x_ijn = log(y_ijn / (1 - y_ijn));  // reverse sigmoid(log==ln)
            float activation_func_derivative = x_ijn * (1 - x_ijn);

            // result
            vals_by_filter[n] += delta * w * activation_func_derivative;
          }
        }

        //
      }
    }

    // write results
    for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
      target[idx + n] = vals_by_filter[n];
    }

    // end
  }
}
