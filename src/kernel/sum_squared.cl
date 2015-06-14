// atomic 64 extension
// see: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/atomicFunctions.html
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/*
 * Part of mean square error calculations. Here we take 2 same sized,
 * single color channel buffers with image data,
 * get the difference between respective pixels, square it and sum.
 *
 *
 * Code partially inspired by:
 * http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
 */

__kernel
void main(__read_only __global uchar* original_image,
          __read_only __global float* algo_result,
          __local float* scratch,
          volatile __global ulong* target,
          __const uint pixel_count){
  // note we operate on floats in local stage
  // an then switch to long for global

  const int global_index = get_global_id(0);
  const int local_index = get_local_id(0);

  // each kernel computes it's value and stores in local scratch buffer
  float squared_diff = 0.0;
  if (global_index < pixel_count) {
    uchar a = original_image[global_index];
    float b = algo_result[global_index];
    float d  = a - b;
    squared_diff = d*d;
  }
  scratch[local_index] = squared_diff;

  // wait till all kernels from local groups finished
  barrier(CLK_LOCAL_MEM_FENCE);

  // add all squared_diffs for local group
  for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
      if (local_index < offset) {
        float other = scratch[local_index + offset];
        float mine = scratch[local_index];
        scratch[local_index] = mine + other;
      }
      // wait for all local kernels to finish previous step and to reach stable state
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  // add local result to global result
  if (local_index == 0) {
    ulong local_result = convert_ulong(scratch[0] + 0.5);
    atomic_add(target, local_result);
  }
}
