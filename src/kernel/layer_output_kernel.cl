/**
 * Special case of layer/forward kernel - when current_filter_count==1
 * NOTE: there is only one bias for a;; kernels, but it's current value
 * is held on GPU so we still have to use pointer instad of register
 *
 * run for global:[ow,oh]
 */
__kernel void forward__last(__read_only __global float* input,    //
                            __global float* target,               //
                            __read_only __global float* weights,  //
                            __read_only __global float* bias,     //
                            __local float* local_bias,            //
                            __local float* local_weights,         //
                            uint input_w, uint input_h) {         //
  const int2 pos = {get_global_id(0), get_global_id(1)};
  const int2 block_pos = {get_local_id(0), get_local_id(1)};
  // const int2 block_dim = {get_local_size(0), get_local_size(1)};
  const int block_size = get_local_size(0) * get_local_size(1);
  const int2 out_size = {input_w - F_SPATIAL_SIZE + 1,
                         input_h - F_SPATIAL_SIZE + 1};
  const int out_idx = (pos.y * out_size.x) + pos.x;

  // copy bias value to local memory for faster access
  if (block_pos.x == 0 && block_pos.y == 0) {
    *local_bias = *bias;
  }

  // copy weights to local memory for faster access
  const int weights_size =
      F_SPATIAL_SIZE * F_SPATIAL_SIZE * PREVIOUS_FILTER_COUNT;
  int iidx = block_pos.y * get_local_size(0) + block_pos.x;
  while (iidx < weights_size) {
    local_weights[iidx] = weights[iidx];
    iidx += block_size;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (pos.x < 0 || pos.x >= out_size.x ||  //
      pos.y < 0 || pos.y >= out_size.y)
    return;

  float sum = *local_bias;
  for (size_t dx = 0; dx < F_SPATIAL_SIZE; dx++) {
    for (size_t dy = 0; dy < F_SPATIAL_SIZE; dy++) {
      int2 input_pos = {pos.x + dx, pos.y + dy};
      int input_idx =
          ((input_pos.y * input_w) + input_pos.x) * PREVIOUS_FILTER_COUNT;
      size_t w_idx = ((dy * F_SPATIAL_SIZE) + dx) * PREVIOUS_FILTER_COUNT;

// following unroll gives ~7% better performance, but it's just so naive..
#pragma unroll 4
      for (size_t k = 0; k < PREVIOUS_FILTER_COUNT; k++) {
        // following line is responsible for bad performance. I'm not sure
        // how to optimize this..
        float input_value = input[input_idx + k];
        float w = local_weights[w_idx + k];
        sum += w * input_value;
      }
    }
  }

#ifdef SKIP_RELU
  target[out_idx] = sum;
#else
  target[out_idx] = max(sum, 0.0f);
#endif  // SKIP_RELU
}
