
/* clang-format off */
/**
 *
 * Calculate deltas*activation_func_derivative of previous layer.
 *
 * In following notation (l), (l-1) describes relative layer and [...] lower indices.
 *
 * Algo for delta_ijn on layer (l-1), where:
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
 *         * delta[i-a,j-b,k](l)      # (2) error term for [i-a,j-b,k](l). minus since we have point (i,j) and we asking: 'which output point are we affecting with w[a,b,_,_]'
 *         * f`(x[i,j,n](l-1) )       # (3) derivative of activation function at measured point
 *
 * TODO in (3) should index be x[i+a,j+b,n] or x[i,j,n]?
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
__kernel void deltas(__read_only __global float* deltas_next_layer,  //
                     __read_only __global float* layer_output,       //
                     __global float* target,                         //
                     __read_only __global float* W,                  //
                     uint f_spatial_size,                            //
                     uint f_next_spatial_size,                       //
                     uint n_next_filter_cnt,                         //
                     uint layer_out_w, uint layer_out_h) {
  // x=col=i; range: 0..layer_out_w
  // y=row=j; range: 0..layer_out_h
  const int2 pos = {get_global_id(0), get_global_id(1)};
  const uint sample_id = get_global_id(2);
  const int2 out_dim = {layer_out_w, layer_out_h};
  const int idx = ((pos.y * out_dim.x) + pos.x) * CURRENT_FILTER_COUNT;
  const int2 next_layer_out = {out_dim.x - f_next_spatial_size + 1,
                               out_dim.y - f_next_spatial_size + 1};

#define IMAGE_OFFSET_CURR \
  sample_id* CURRENT_FILTER_COUNT* layer_out_w* layer_out_h
#define IMAGE_OFFSET_NEXT \
  sample_id* n_next_filter_cnt* next_layer_out.x* next_layer_out.y

  // zeroed result cache and read read output values for output[i,j,n]
  float delta_for_filter[CURRENT_FILTER_COUNT];
  float activation_func_derivatives[CURRENT_FILTER_COUNT];

  // range check for i,j
  if (pos.x >= 0 && pos.x < out_dim.x &&  //
      pos.y >= 0 && pos.y < out_dim.y) {
    // fill tmp buffer values
    for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
      delta_for_filter[n] = 0.0f;
      // (3) f`( x[i,j,n](l-1) )
      float y_ijn = layer_output[IMAGE_OFFSET_CURR + idx + n];
      activation_func_derivatives[n] = y_ijn > 0.0f ? 1.0f : 0.0f;
    }

    for (size_t dy = 0; dy < f_next_spatial_size; dy++) {
      for (size_t dx = 0; dx < f_next_spatial_size; dx++) {
        // NOTE: dy=a, dx=b
        int2 next_layer_pos = {pos.x - dx, pos.y - dy};
        size_t w_idx_2D = ((dy * f_next_spatial_size) + dx) *
                          n_next_filter_cnt * CURRENT_FILTER_COUNT;

        for (size_t k = 0; k < n_next_filter_cnt; k++) {
          // (2) delta[i+a,j+b,k](l)
          // this requires us to map curent output_pos to next layer coords,
          // but some of the point may not be in range. f.e. point(i=0,j=0) does
          // not affect output with w[a,b] if a!=0 && b!=0
          int next_layer_idx =
              ((next_layer_pos.y * next_layer_out.x) + next_layer_pos.x) *
              n_next_filter_cnt;
          bool in_range =
              next_layer_pos.x >= 0 && next_layer_pos.x < next_layer_out.x &&
              next_layer_pos.y >= 0 && next_layer_pos.y < next_layer_out.y;
          float delta =
              in_range
                  ? deltas_next_layer[IMAGE_OFFSET_NEXT + next_layer_idx + k]
                  : 0.0f;

          for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
            // (1) w[abnk](l-1)
            // NOTE: n iterates over lower layer's filters
            size_t w_idx = w_idx_2D + n * n_next_filter_cnt + k;
            float w = W[w_idx];

            // (3) f`( x[i,j,n](l-1) )
            float activation_func_derivative = activation_func_derivatives[n];

            // result
            delta_for_filter[n] += delta * w * activation_func_derivative;
          }
        }

        //
      }
    }

    // write results
    for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
      target[IMAGE_OFFSET_CURR + idx + n] = delta_for_filter[n];
    }

    // end
  }
}
