
/* clang-format off */
/**
 *
 * Calculate grad_w and grad_b. Requires additional results summation.
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
 * 	CURRENT_FILTER_COUNT                     filter_count(l)
 *  PER_FILTER_SIZE                          f_spatial_size * f_spatial_size * n_prev_filter_cnt
 *
 * @param  float*      deltas                size: output_w(l) * output_w(l) * filter_count(l)
 * @param  float*      layer_input           size: output_w(l-1) * output_w(l-1) * filter_count(l-1)
 * @param  float*      target_grad_w         size: num_of_all_local_groups * (PER_FILTER_SIZE * CURRENT_FILTER_COUNT)
 * @param  float*      target_grad_b         size: num_of_all_local_groups * CURRENT_FILTER_COUNT
 * @param  float*      scratch_w             size: local_size * PER_FILTER_SIZE
 * @param  float*      scratch_b             size: local_size * CURRENT_FILTER_COUNT
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
                   __local float* scratch_w,                 //
                   __local float* scratch_b,                 //
                   uint n_prev_filter_cnt,                   //
                   uint f_spatial_size,                      //
                   uint layer_out_w, uint layer_out_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const int2 out_size = {layer_out_w, layer_out_h};
  /* clang-format off */
  const size_t local_index = get_local_id(0) * get_local_size(1) + get_local_id(1);    // id in group
  const size_t local_size = get_local_size(0) * get_local_size(1) + get_local_size(1); // work items in group
  const size_t group_id = get_group_id(0) * get_num_groups(1) + get_group_id(1);       // id of group
  const size_t numbers_per_w = PER_FILTER_SIZE * CURRENT_FILTER_COUNT; // number of floats in w[_,_,_,_]
  /* clang-format on */

  const int idx = ((pos.y * layer_out_w) + pos.x) * CURRENT_FILTER_COUNT;
  const uint input_w = layer_out_w + f_spatial_size - 1;

  // buffer that we are going to fill CURRENT_FILTER_COUNT times
  // represents values per a*b*k in w[abnk]
  float tmp_buffer[PER_FILTER_SIZE];

  // range check for i,j
  const bool in_range = pos.x >= 0 && pos.x < out_size.x &&  //
                        pos.y >= 0 && pos.y < out_size.y;

  for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
    // we are doing calculations on per filter basis.
    // expect memory bariers inside current `for.
    // All this so that we can reduce total global memory usage.
    // (If we are using local barriers all work items in group MUST reach
    // barrier, so we cannot early bail if (i,j) is not in range)

    // reset tmp_buffer
    for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
      tmp_buffer[i] = 0.0f;
    }
    float grad_bias = 0.0f;

    // fill tmp_buffer with values if we are in range, else it will stay at
    // zeroes, which suits us perfectly
    if (in_range) {
      // (1) delta[i,j,n](l)
      // we have all of i,j,n at this point
      float delta = deltas[idx + n];
      grad_bias = delta;

      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          // TODO this indexing does not depend on n, move out of the loop
          size_t tmp_idx = ((dy * f_spatial_size) + dx) * n_prev_filter_cnt;
          // index on previous layer
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

    // now, for filter n we have folowing partial results for point (i,j):
    // - tmp_buffer has grad_w[_,_,n,_]
    // - grad_bias has bias gradient for filter n
    // According to algoritm we have to add all partial results (for all i,j)
    // We will start with adding values from local group

    // copy local results to locally accesible memory
    size_t local_offset_w = local_index * PER_FILTER_SIZE;
    for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
      scratch_w[local_offset_w + i] = tmp_buffer[i];
    }
    scratch_b[local_index] = grad_bias;

    // wait till all kernels from local group finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // add all for local group
    // (for simpler version of this method see sum_kernel)
    for (int offset = local_size / 2; offset > 0; offset = offset / 2) {
      if (local_index < offset) {
        // weights
        for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
          size_t idx1 = local_offset_w + i;
          size_t idx2 = (local_index + offset) * PER_FILTER_SIZE + i;
          scratch_w[idx1] = scratch_w[idx1] + scratch_w[idx2];
        }
        // bias
        float other = scratch_b[local_index + offset];
        float mine = scratch_b[local_index];
        scratch_b[local_index] = mine + other;
      }

      // wait for all local kernels to finish step and reach stable state
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write local result to global result
    // (since scrath_w holds only grad_w[_,_,n,_], we have to manually
    // reconstruct the offset)
    if (local_index == 0) {
      size_t global_grad_w_offset = group_id * numbers_per_w;
      // weights
      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          size_t tmp_idx = ((dy * f_spatial_size) + dx) * n_prev_filter_cnt;
          size_t w_idx_2D = ((dy * f_spatial_size) + dx) *
                            CURRENT_FILTER_COUNT * n_prev_filter_cnt;
          for (size_t k = 0; k < n_prev_filter_cnt; k++) {
            float grad_val = scratch_w[tmp_idx + k];
            // dy=a, dx=b, we also have n, k <- all that is needed to index
            // w[abnk]
            size_t w_idx = w_idx_2D + k * CURRENT_FILTER_COUNT + n;
            target_grad_w[global_grad_w_offset + w_idx] = grad_val;
          }
        }
      }
      // bias
      target_grad_b[group_id] = scratch_b[0];
    }

    // end
  }
}
