#ifndef __KERNEL_H
#define __KERNEL_H

#include "CL/opencl.h"

namespace opencl {

// forward declaration
class Context;
struct MemoryHandler;

class Kernel {
 public:
  void init(Context *, cl_kernel, cl_program, size_t max_work_group_size);
  void cleanup();

  /**
   * Set the next argument. To be used as a sequence of calls,
   * where each one sets next argument.
   *
   * @param arg_size  size of pointer f.e. sizeof(cl_mem) | sizeof(cl_int)
   * @param arg_value void* pointer to argument value
   */
  void push_arg(size_t arg_size, const void *);

  /**
   * Set the next argument. To be used as a sequence of calls,
   * where each one sets next argument.
   *
   * @param handle  gpu memory handler
   */
  void push_arg(const MemoryHandler *);

  /**
   * Execute the kernel with arguments that were pushed before this call.
   * After this call You will have to provide all arguments againg before
   * You execute the kernel again.
   * Also this function provides some basics checks for work_size parameters,
   * so You can catch them more easily.
   *
   * @param  work_dim                 number of dimensions
   * @param  global_work_size         :size_t*, total work size provided as
   *array
   *each value for one of dimensions
   * @param  local_work_size          :size_t*, work group size
   * @param  events_to_wait_for       [OPT] wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event execute(cl_uint work_dim,                //
                   const size_t *global_work_size,  //
                   const size_t *local_work_size,   //
                   cl_event *events_to_wait_for = nullptr, int event_count = 0);

  inline size_t get_max_work_group_size() const { return max_work_group_size; }
  inline Context *get_context() const { return context; }

 private:
  /**
   * Basic checks for work parameters. Based on:
   * https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clEnqueueNDRangeKernel.html
   *
   * @return                  if work parameters fulfill the constraints
   */
  void check_work_parameters(cl_uint work_dim,  //
                             const size_t *global_work_size,
                             const size_t *local_work_size);

 private:
  cl_kernel kernel_id;
  cl_program program_id;
  Context *context;
  size_t max_work_group_size;
  int arg_stack_size;
  bool initialized = false;
};

//
}

#endif /* __KERNEL_H   */
