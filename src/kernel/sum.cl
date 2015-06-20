// atomic 64 extension
// see: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/atomicFunctions.html
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/*
 * Code partially inspired by:
 * http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
 */

__kernel
void main(__read_only __global float* data,
          volatile __global ulong* target,
          __local float* scratch,
          __const uint len){
  // note we operate on floats in local stage
  // an then switch to long for global

  const int global_index = get_global_id(0);
  const int local_index = get_local_id(0);

  // each kernel computes it's value and stores in local scratch buffer
  scratch[local_index] = global_index < len ? data[global_index] : 0.0;

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
