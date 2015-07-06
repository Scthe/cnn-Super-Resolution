
/* clang-format off */
/**
 *
 * In following notation (l), (l-1) describes relative layer and [...] lower indices.
 *
 * Algo for dJ/dw[abnk] on layer (l), where:
 *   a = 0..spatial_size(l)
 *   b = 0..spatial_size(l)
 *   n = 0..filter_count(l)
 *   k = 0..filter_count(l+1)
 *
 * dJ/dw[abnk] = 0
 * for i = 0..output_w(l):               # for each node where this weight is used
 * for j = 0..output_h(l):
 *   for n = 0..filter_count(l):
 *     for a = 0..spatial_size(l):       # offset to weight
 *     for b = 0..spatial_size(l):       # (it's kernel size, what You expect ?)
 *       for k = 0..filter_count(l-1):   # for all inputs
 *         dJ/dw[abnk] += deltas[i,j,n]  # (1) error for this point
 *           * layer_input[i+a,j+b,k]    # (2) input at this point
 *
 * macros:
 * 	CURRENT_FILTER_COUNT                     filter_count(l-1)
 *  PER_FILTER_SIZE                          f_spatial_size * f_spatial_size * n_prev_filter_cnt
 *
 * @param  float*      deltas                size: output_w(l) * output_w(l) * filter_count(l)
 * @param  float*      layer_input           size: output_w(l-1) * output_w(l-1) * filter_count(l-1)
 * @param  float*      target_grad_w         size: num_of_all_local_groups *
 *                                           (f_spatial_size
 *                                            * f_spatial_size
 *                                            * CURRENT_FILTER_COUNT
 *                                            * n_prev_filter_cnt)
 * @param  float*      target_grad_b         size: num_of_all_local_groups * CURRENT_FILTER_COUNT
 * @param  float*      per_filter_scratch_w  size: local_size * PER_FILTER_SIZE
 * @param  float*      per_filter_scratch_b  size: local_size * CURRENT_FILTER_COUNT
 * @param  uint        n_prev_filter_cnt     spatial/kernel size for (l-1)
 * @param  uint        f_spatial_size        spatial/kernel size for (l)
 * @param  uint        layer_out_w           output_w(l)
 * @param  uint        layer_out_h           output_h(l)
 * @return {[type]}                          [description]
 */
/* clang-format on */
__kernel void main(__read_only __global float* deltas,       //
                   __read_only __global float* layer_input,  //
                   __global float* target_grad_w,            //
                   __global float* target_grad_b,            //
                   __local float* per_filter_scratch_w,      //
                   __local float* per_filter_scratch_b,      //
                   uint n_prev_filter_cnt,                   //
                   uint f_spatial_size,                      //
                   uint layer_out_w, uint layer_out_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const int2 out_size = {layer_out_w, layer_out_h};
  /* clang-format off */
  const size_t local_index = get_local_id(0) * get_local_size(1) + get_local_id(1);
  const size_t local_size = get_local_size(0) * get_local_size(1) + get_local_size(1);
  const size_t numbers_per_w = f_spatial_size * f_spatial_size * CURRENT_FILTER_COUNT * n_prev_filter_cnt;
  const size_t group_id = get_group_id(0) * get_num_groups(1) + get_group_id(1);
  /* clang-format on */

  const int base_point_idx =
      ((pos.y * layer_out_w) + pos.x) * CURRENT_FILTER_COUNT;
  const uint input_w = layer_out_w + f_spatial_size - 1;

  // buffer that we are going to fill CURRENT_FILTER_COUNT times
  // represents a*b*k in w[abnk]
  float tmp_buffer[PER_FILTER_SIZE];

  // range check for i,j
  const bool in_range = pos.x >= 0 && pos.x < out_size.x &&  //
                        pos.y >= 0 && pos.y < out_size.y;

  for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
    // we are doing calculations on per filter basis.
    // expect memory bariers inside current `for.
    // All this so that for whole local group it will similiar instruction set.

    // reset tmp_buffer
    for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
      tmp_buffer[i] = 0.0f;
    }
    float grad_bias = 0.0f;

    // if we are in range the fill tmp_buffer with values, else it will stay at
    // zeroes.
    if (in_range) {
      // (1) delta[i,j,n](l)
      // we have all of i,j,n at this point
      float delta = deltas[base_point_idx + n];
      grad_bias = delta;

      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          size_t tmp_idx = ((dy * f_spatial_size) + dx) * n_prev_filter_cnt;
          int2 prev_layer_pos = {pos.x + dx, pos.y + dy};
          int prev_layer_idx =
              ((prev_layer_pos.y * input_w) + prev_layer_pos.x) *
              n_prev_filter_cnt;

          for (size_t k = 0; k < n_prev_filter_cnt; k++) {
            // result = delta * {(2) input values}
            tmp_buffer[tmp_idx + k] = delta * layer_input[prev_layer_idx + k];
          }
        }
        //
      }
    }

    // now tmp_buffer has been filler with grad_w for filter n
    // and grad_bias holds delta for filter n
    // We have to add respective values for all kernels. We will start with
    // adding values from local group
    for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
      per_filter_scratch_w[local_index * PER_FILTER_SIZE + i] = tmp_buffer[i];
    }
    per_filter_scratch_b[local_index] = grad_bias;

    // wait till all kernels from local group finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // add all for local group
    for (int offset = local_size / 2; offset > 0; offset = offset / 2) {
      if (local_index < offset) {
        // weights
        for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
          size_t idx1 = local_index * PER_FILTER_SIZE + i;
          size_t idx2 = (local_index + offset) * PER_FILTER_SIZE + i;
          per_filter_scratch_w[local_index] =
              per_filter_scratch_w[idx1] + per_filter_scratch_w[idx2];
        }
        // bias
        float other = per_filter_scratch_b[local_index + offset];
        float mine = per_filter_scratch_b[local_index];
        per_filter_scratch_b[local_index] = mine + other;
      }
      // wait for all local kernels to finish step and reach stable state
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write local result to global result
    if (local_index == 0) {
      // weights
      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          size_t tmp_idx = ((dy * f_spatial_size) + dx) * n_prev_filter_cnt;
          size_t w_idx_2D = ((dy * f_spatial_size) + dx) *
                            CURRENT_FILTER_COUNT * n_prev_filter_cnt;
          for (size_t k = 0; k < n_prev_filter_cnt; k++) {
            float grad_val = per_filter_scratch_w[tmp_idx + k];
            // dy=a, dx=b, we also have n, k <- all that is needed to index
            // w[abnk]
            size_t w_idx = w_idx_2D + k * CURRENT_FILTER_COUNT + n;
            w_idx += group_id * numbers_per_w;
            target_grad_w[w_idx] = grad_val;
          }
        }
      }

      // bias
      target_grad_b[group_id] = per_filter_scratch_b[0];
    }

    // end
  }
}
