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

/**
 * Part of mean square error calculations. Here we take 2 same sized,
 * single color channel buffers with image data and get the difference
 * between respective pixels.
 */
__kernel void squared_err(__read_only __global float* ground_truth_image,
                          __read_only __global float* algo_result,
                          __global float* target,       //
                          __local float* scratch,       //
                          __const uint ground_truth_w,  //
                          __const uint ground_truth_h,  //
                          __const uint algo_result_w,   //
                          __const uint algo_result_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const uint sample_id = get_global_id(2);
  const int2 out_size = {algo_result_w, algo_result_h};
  const int idx = (pos.y * algo_result_w) + pos.x;
  const size_t padding = (ground_truth_w - algo_result_w) / 2;
  const size_t local_size = get_local_size(1) * get_local_size(0),
               local_index =
                   get_local_id(1) * get_local_size(0) + get_local_id(0);

#define IMAGE_OFFSET_GT sample_id* ground_truth_w* ground_truth_h
#define IMAGE_OFFSET_ALGO sample_id* algo_result_w* algo_result_h

  // size of ground_truth != algo res (padding)
  // The offset is not const, since it depends on the row we are in
  // algo for ground_truth_idx:
  // (row + padding_on_top_of_image) * width + padding_left + col
  const size_t ground_truth_idx =
      (pos.y + padding) * ground_truth_w + padding + pos.x;

  float squared_diff = 0.0f;
  if (pos.x >= 0 && pos.x < out_size.x &&  //
      pos.y >= 0 && pos.y < out_size.y) {
    float t = ground_truth_image[IMAGE_OFFSET_GT + ground_truth_idx];
    float y = algo_result[IMAGE_OFFSET_ALGO + idx];
    float d = y - t;
    squared_diff = d * d;
  }
  scratch[local_index] = squared_diff;

  // wait till all kernels from local groups finished
  barrier(CLK_LOCAL_MEM_FENCE);

  // add all squared_diffs for local group
  for (int offset = local_size / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = mine + other;
    }
    // wait for all local kernels to finish previous step
    // and reach stable state
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // add local result to global result
  if (local_index == 0) {
    atomic_add_global(target, scratch[0]);
  }
}
