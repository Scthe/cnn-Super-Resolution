/* clang-format off */
/**
 * @see http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
 * @see  http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
 * 
 * @param {[type]} volatile __global float   *source       [description]
 * @param {[type]} const    float    operand [description]
 */
inline void atomic_add_global(volatile __global float* source, const float operand) {
  /* clang-format on */
  union {
    unsigned int intVal;
    float floatVal;
  } newVal;

  union {
    unsigned int intVal;
    float floatVal;
  } prevVal;

  // NOTE: atomic_cmpxchg(volatile __global unsigned int *p,
  // 	                    unsigned int cmp, unsigned int val)
  do {
    prevVal.floatVal = *source;
    newVal.floatVal = prevVal.floatVal + operand;
  } while (atomic_cmpxchg((volatile __global unsigned int*)source,
                          prevVal.intVal,  //
                          newVal.intVal) != prevVal.intVal);
}

/* clang-format off */
/**
 *
 * Calculate grad_w and grad_b. Requires additional results summation.
 *
 * TODO use atomic floats as in:

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
  const size_t local_index = get_local_id(1) * get_local_size(0) + get_local_id(0);    // id in group
  const size_t local_size = get_local_size(1) * get_local_size(0); // work items in group
  const size_t group_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);       // id of group
  const size_t numbers_per_w = PER_FILTER_SIZE * CURRENT_FILTER_COUNT; // number of floats in w[_,_,_,_]
  const size_t local_offset_w = local_index * PER_FILTER_SIZE;
  /* clang-format on */

  const int idx = ((pos.y * layer_out_w) + pos.x) * CURRENT_FILTER_COUNT;
  const uint input_w = layer_out_w + f_spatial_size - 1;

  // range check for i,j
  const bool in_range = pos.x >= 0 && pos.x < out_size.x &&  //
                        pos.y >= 0 && pos.y < out_size.y;

  for (size_t n = 0; n < CURRENT_FILTER_COUNT; n++) {
    // we are doing calculations on per filter basis.
    // expect memory bariers inside current `for.
    // All this so that we can reduce total global memory usage.
    // (If we are using local barriers all work items in group MUST reach
    // barrier, so we cannot early bail if (i,j) is not in range)
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
            // TODO writing to local memory in loop
            size_t idx = local_offset_w + tmp_idx + k;
            scratch_w[idx] = delta * layer_input[prev_layer_idx + k];
          }
        }
        //
      }
    }
    scratch_b[local_index] = grad_bias;

    /*
    // TODO remove tmp - test use only
    size_t iidx = local_index * CURRENT_FILTER_COUNT + n;
    // target_grad_w[local_index] = local_index;
    if (in_range) {
      target_grad_w[iidx] =  //
                             // idx / CURRENT_FILTER_COUNT;
          // layer_out_w;
          //  grad_bias;
          // pos.y;
          local_size;
    } else {
      target_grad_w[local_index * CURRENT_FILTER_COUNT + n] = -99;
    }
    */

    // We have both grad_w[_,_,n,_] and bias for point (i,j)
    // According to algoritm we have to add all partial results (for all i,j)
    // We will start with adding values from local group

    // wait till all kernels from local group finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // add all for local group
    // (for simpler version of this method see sum_kernel)
    for (int offset = local_size / 2; offset > 0; offset = offset / 2) {
      if (local_index < offset) {
        // weights
        // for (size_t i = 0; i < PER_FILTER_SIZE; i++) {
        // size_t idx1 = local_offset_w + i;
        // size_t idx2 = (local_index + offset) * PER_FILTER_SIZE + i;
        // scratch_w[idx1] = scratch_w[idx1] + scratch_w[idx2];
        // }
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
      // NOTE: atomic_add_global is custom function, see beginning of the file
      // weights
      /*
      for (size_t dy = 0; dy < f_spatial_size; dy++) {
        for (size_t dx = 0; dx < f_spatial_size; dx++) {
          size_t tmp_idx = ((dy * f_spatial_size) + dx) * n_prev_filter_cnt;
          size_t w_idx_2D = ((dy * f_spatial_size) + dx) *
                            CURRENT_FILTER_COUNT * n_prev_filter_cnt;
          for (size_t k = 0; k < n_prev_filter_cnt; k++) {
            float grad_val = scratch_w[tmp_idx + k];
            size_t w_idx = w_idx_2D + k * CURRENT_FILTER_COUNT + n;
            target_grad_w[global_grad_w_offset + w_idx] = grad_val;
          }
        }
      }
      */
      // bias
      atomic_add_global(target_grad_b + n, scratch_b[0]);
    }

    // end
  }
}
