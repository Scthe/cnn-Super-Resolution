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
 * Code partially inspired by:
 * http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
 */
__kernel void sum(__read_only __global float* data,  //
                  volatile __global float* target,   //
                  __local float* scratch,            //
                  __const uint len) {
  const int global_index = get_global_id(0);
  const int local_index = get_local_id(0);

  // each kernel computes it's value and stores in local scratch buffer
  float val = global_index < len ? data[global_index] : 0.0f;
#ifdef SUM_SQUARED
  val = val * val;
#endif
  scratch[local_index] = val;

  // wait till all kernels from local groups finished
  barrier(CLK_LOCAL_MEM_FENCE);

  // add all squared_diffs for local group
  for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
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
