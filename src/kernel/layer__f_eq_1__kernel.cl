/**
 * Special case of layer/forward kernel - when kernel's spatial size==1
 *
 * run for global:[ow,oh,current_filter_count], local:[_,_,current_filter_count]
 * NOTE: f=1, so no range check needed, everything is greatly simplified
 */
__kernel void main(__read_only __global float* input,                  //
                   __global float* target,                             //
                   __read_only __global float* weights,                //
                   __read_only __global float* bias,                   //
                   uint n_prev_filter_cnt, uint current_filter_count,  //
                   uint f_spatial_size,                                //
                   uint input_w, uint input_h) {
  const int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
  const int n = pos.z;
  const size_t out_idx = (pos.y * input_w + pos.x) * current_filter_count + n,
               base_input_idx = (pos.y * input_w + pos.x) * n_prev_filter_cnt;

  // copy input for this position to local memory
  // TODO fix this - returns NAN
  /*
  __local float prev_filter_vals[PREVIOUS_FILTER_COUNT];
  int kk = pos.z;
  while (kk < PREVIOUS_FILTER_COUNT) {
    prev_filter_vals[kk] = input[base_input_idx + kk];
    kk += PREVIOUS_FILTER_COUNT;
  }
  */
  barrier(CLK_LOCAL_MEM_FENCE);
  if (pos.x < 0 || pos.x >= input_w ||  //
      pos.y < 0 || pos.y >= input_h ||  //
      pos.z < 0 || pos.z >= current_filter_count)
    return;

  float sum = bias[n];
  for (size_t k = 0; k < n_prev_filter_cnt; k++) {
    float point_value = input[base_input_idx + k];
    // float point_value = prev_filter_vals[k];
    float w = weights[n + k * current_filter_count];
    float result = point_value * w;
    sum += result;
  }

#ifdef SKIP_RELU
  target[out_idx] = sum;
#else
  target[out_idx] = max(sum, 0.0f);
#endif  // SKIP_RELU
}
