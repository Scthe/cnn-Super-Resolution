#ifndef KERNEL_H
#define KERNEL_H

#include "CL/opencl.h"
#include <iostream>  // for std::ostream& operator<<(..)

#define MAX_KERNEL_IDENTIFIER_SIZE 128

namespace opencl {

// forward declaration
class Context;
typedef size_t MemoryHandle;

class Kernel {
 public:
  void init(Context *, cl_kernel, cl_program,  //
            const char *, const char *);
  void cleanup();
  friend std::ostream &operator<<(std::ostream &os, opencl::Kernel &p);

  size_t current_local_memory();

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
  void push_arg(MemoryHandle);

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
  inline cl_ulong get_total_execution_time() const {
    return execution_time_sum;
  }
  inline const char *get_human_identifier() const { return human_identifier; }

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
  cl_ulong private_mem_size;
  size_t pref_work_group_multiple;

  size_t arg_stack_size;
  size_t assigned_local_memory;  // by hand, since it does always work
  bool initialized = false;

  /** meaningful only if context->is_running_profile_mode */
  cl_ulong execution_time_sum = 0;
  char human_identifier[MAX_KERNEL_IDENTIFIER_SIZE];
};

//
}

// std::ostream &operator<<(std::ostream &, opencl::Kernel &);

#endif /* KERNEL_H   */
