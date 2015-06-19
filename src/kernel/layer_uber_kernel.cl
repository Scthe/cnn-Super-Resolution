float sigmoid(float x){
  return 1 / (1 + exp(-x));
}

// MACRO: CURRENT_FILTER_COUNT filter count for curernt layer
// if CURRENT_FILTER_COUNT is not defined it is assumed to be 1
// TODO: optimize the code adding ifdef checks

// MACRO: RESULT_MULTIPLY if defined: sigmoid will not be aplied to the result.
// Instead it will be multiplied by RESULT_MULTIPLY value.

/**
*
* @param source               output of previous layer, size:
*                               * 1st layer: img_w * img_h
*                               * 2nd layer: (img_w-f1+1) * (img_h-f1+1) * n1
*                               * 3rd layer: (img_w-f1-f2+2) * (img_h-f1-f2+2) * n2 // TODO check
* @param target               zeroed output buffer, size:
*                               * 1st layer: (img_w-f1+1) * (img_h-f1+1) * n1
*                               * 2nd layer: (img_w-f1-f2+2) * (img_h-f1-f2+2) * n2
*                               * 3rd layer: (img_w-f1-f2-f3+3) * (img_h-f1-f2-f3+3)
* @param W                    weights, size:
*                                * 1st layer: f1*f1 per each filter (total: f1*f1*n1)
*                                * 2nd layer: f2*f2*n1 per each filter (total: f2*f2*n1*n2)
*                                * 3rd layer: f3*f3*n2
* @param B                    biases, size:
*                                * 1st layer: n1
*                                * 2nd layer: n2
*                                * 3rd layer: 1
* @param n_prev_filter_cnt    1/n1/n2
* @param f_spatial_size       current: f1/f2/f3
* @param src_w                source width
* @param src_h                source height
*/
__kernel
void main(__read_only __global float* source,
          __global float* target,
          __read_only __global float* W,
          __read_only __global float* B,
          uint n_prev_filter_cnt,
          uint f_spatial_size,
          uint src_w, uint src_h){

  // value range: (0..out_w, 0..out_h)
  const int2 pos = {get_global_id(0), get_global_id(1)};

  // const int2 out_size = {out_w, out_h};
  const int2 src_size = {src_w, src_h};
  const int2 out_size = { src_w - f_spatial_size + 1,
                          src_h - f_spatial_size + 1};

  // left top corner of source
  // const int2 padding = {f_prev_spatial_size / 2,
                        // f_prev_spatial_size / 2};
  // due too spatial size != 0 we cant calcutate values for points that are on the edge of source
  // const int2 pos_src = pos + padding;

  // index on which write to target, will write total of CURRENT_FILTER_COUNT values
  const int out_idx = ((pos.y * out_size.x) + pos.x) * CURRENT_FILTER_COUNT;

  // when working with source with at least 3 dimensions we have
  // to multiply by src_h for indexing purposes.
  // It's just 1 when working with 2D source.
  const int third_dimension_factor_weigth = (n_prev_filter_cnt > 1)? f_spatial_size : 1;

  // value range check
  if(pos.x >= 0 && pos.x <= out_size.x &&
     pos.y >= 0 && pos.y <= out_size.y){

      // zeroed result cache
      float vals_by_filter[CURRENT_FILTER_COUNT];
      for (size_t filter_id = 0; filter_id < CURRENT_FILTER_COUNT; filter_id++) {
        vals_by_filter[filter_id] = 0.0f;
      }

      // apply weights
      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          int2 delta = {dx, dy};
          int2 point_pos = pos + delta;
          // int base_point_idx  = ((point_pos.y * src_w) + point_pos.x) * src_h;
          int base_point_idx  = ((point_pos.y * src_w) + point_pos.x) * n_prev_filter_cnt; // TODO * src_h or * n_prev_filter_cnt?
          int base_W_idx = ((dy * f_spatial_size) + dx) * third_dimension_factor_weigth;

          for (size_t i_n = 0; i_n < n_prev_filter_cnt; i_n++) {
            // for every feature map in source (only 1 in 2D):
            float point_value = source[base_point_idx + i_n];
            // offset by old_filter_i and use this as all filter values
            // that are 'before' in W array (this op. is quite weird with indices)
            int base_W_idx2 = (base_W_idx + i_n) * CURRENT_FILTER_COUNT;

            for (size_t filter_id = 0; filter_id < CURRENT_FILTER_COUNT; filter_id++) {
              // do it for all fitlers in this layer:
              float W_value = W[base_W_idx2 + filter_id];
              vals_by_filter[filter_id] += W_value * point_value;
            }
          }
        }
      }

      // add bias
      // write cached results to target buffer
      for (size_t filter_id = 0; filter_id < CURRENT_FILTER_COUNT; filter_id++) {
        float B_value = B[filter_id];
        float result = vals_by_filter[filter_id] + B_value;

#ifndef RESULT_MULTIPLY
        target[out_idx + filter_id] = sigmoid(result);
#else
        target[out_idx + filter_id] = result * RESULT_MULTIPLY;
#endif // RESULT_MULTIPLY

      }

  }
}
