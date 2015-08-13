#include "Kernel.hpp"
#include "Context.hpp"

#include <iostream>
#include <cstdio>

namespace opencl {

void Kernel::init(Context *ctx, cl_kernel k, cl_program p, const char *file,
                  const char *args) {
  if (initialized) cleanup();
  this->context = ctx;
  this->kernel_id = k;
  this->program_id = p;
  arg_stack_size = 0;
  assigned_local_memory = 0;
  initialized = true;
  // read parameters
  cl_int ciErr1;
  ciErr1 = clGetKernelWorkGroupInfo(k, context->device().device_id,
                                    CL_KERNEL_WORK_GROUP_SIZE, 1024,
                                    &max_work_group_size, nullptr);
  ciErr1 = clGetKernelWorkGroupInfo(k, context->device().device_id,
                                    CL_KERNEL_PRIVATE_MEM_SIZE, 1024,
                                    &private_mem_size, nullptr);
  ciErr1 =
      clGetKernelWorkGroupInfo(k, context->device().device_id,
                               CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                               1024, &pref_work_group_multiple, nullptr);
  context->check_error(ciErr1, "Could not get kernel informations");

  file = file == nullptr ? "??" : file;
  args = args == nullptr ? "--" : args;
  if (file != nullptr && args != nullptr) {
    snprintf(this->human_identifier, MAX_KERNEL_IDENTIFIER_SIZE, "'%s'[%s]",
             file, args);
  }
}

void Kernel::cleanup() {
  if (!initialized) return;
  initialized = false;

  if (kernel_id) clReleaseKernel(kernel_id);
  if (program_id) clReleaseProgram(program_id);
}

size_t Kernel::current_local_memory() {
  cl_ulong loc_mem_size;
  cl_int ciErr1 = clGetKernelWorkGroupInfo(
      kernel_id, context->device().device_id, CL_KERNEL_LOCAL_MEM_SIZE, 1024,
      &loc_mem_size, nullptr);
  context->check_error(ciErr1, "Could not get kernel's local memory usage");
  return loc_mem_size > assigned_local_memory ? loc_mem_size
                                              : assigned_local_memory;
}

void Kernel::push_arg(size_t arg_size, const void *arg_value) {
  cl_int ciErr1 =
      clSetKernelArg(kernel_id, arg_stack_size, arg_size, arg_value);
  context->check_error(ciErr1, "Could not push kernel argument");
  ++arg_stack_size;
  // local memory
  if (!arg_value) assigned_local_memory += arg_size;
}

void Kernel::push_arg(MemoryHandle gpu_buf) {
  auto mem = context->raw_memory(gpu_buf);
  this->push_arg(sizeof(cl_mem), (void *)&mem->handle);
}

cl_event Kernel::execute(cl_uint work_dim,                //
                         const size_t *global_work_size,  //
                         const size_t *local_work_size,   //
                         cl_event *events_to_wait_for,
                         int events_to_wait_for_count) {
  context->check_error(context->is_initialized(),
                       "Context was not initialized");
  check_work_parameters(work_dim, global_work_size, local_work_size);

  // check used amount of local memory
  char msg_buffer[192];
  auto used_loc_mem = current_local_memory();
  if (used_loc_mem > context->device().local_mem_size) {
    snprintf(msg_buffer, sizeof(msg_buffer),
             "You are using too much local memory(%d), only %llu is available",
             used_loc_mem, context->device().local_mem_size);
    context->check_error(false, msg_buffer);
  }

  // correct event parameters
  if (!events_to_wait_for) events_to_wait_for_count = 0;
  if (events_to_wait_for_count <= 0) events_to_wait_for = nullptr;

  arg_stack_size = 0;  // prepare for next invoke
  assigned_local_memory = 0;
  cl_command_queue *cmd_queue = context->command_queue();

  cl_event finish_token;
  cl_int ciErr1 = clEnqueueNDRangeKernel(
      *cmd_queue, kernel_id,              // what and where to execute
      work_dim, nullptr,                  // must be NULL
      global_work_size, local_work_size,  //
      events_to_wait_for_count, events_to_wait_for,  // sync events
      &finish_token);
  context->check_error(ciErr1, "Error in clEnqueueNDRangeKernel");

  if (context->is_running_profile_mode()) {
    clWaitForEvents(1, &finish_token);
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(finish_token, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(finish_token, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &end, NULL);
    execution_time_sum += (end - start);
  }

  return finish_token;
}

void Kernel::check_work_parameters(cl_uint work_dim,  //
                                   const size_t *global_work_size,
                                   const size_t *local_work_size) {
  // std::cout << std::endl
  // << "Work size: " << ((unsigned int)work_dim)
  // << "/" << (*global_work_size)
  // << "/" << (*local_work_size) << std::endl;

  char msg_buffer[192];
  if (work_dim < 1 || work_dim > 3) {
    snprintf(msg_buffer, sizeof(msg_buffer),
             "Work parameters: 1 <= (work_dim=%d) <= 3", work_dim);
    context->check_error(false, msg_buffer);
  }

  auto device = context->device();
  long long device_work_id_range = ((long long)1) << device.address_bits;
  long long real_global_work_size = 1,
            real_local_work_size = 1;  // # of work-items in work-group
  bool local_dims_lte_device_max = true,
       global_dims_divisible_by_local_dims = true;

  for (size_t i = 0; i < work_dim; i++) {
    real_global_work_size *= global_work_size[i];
    if (local_work_size) {
      real_local_work_size *= local_work_size[i];
      local_dims_lte_device_max &=
          local_work_size[i] <= device.work_items_for_dims[i];
      global_dims_divisible_by_local_dims &=
          global_work_size[i] % local_work_size[i] == 0;
    }
  }

#define WORK_DIMENSIONS_STR "global:[%d,%d,%d], local:[%d,%d,%d]"
#define WORK_DIMENSIONS_VAL global_work_size[0],                     \
                           (work_dim > 1 ? global_work_size[1] : 1), \
                           (work_dim == 3 ? global_work_size[2] : 1),\
                           local_work_size[0],                       \
                           (work_dim > 1 ? local_work_size[1] : 1),  \
                           (work_dim == 3 ? local_work_size[2] : 1)

  bool is_ok = true;
  if (!local_dims_lte_device_max) {
    is_ok = false;
    snprintf(msg_buffer, sizeof(msg_buffer),
             "Work parameters: one of local dimensions are bigger "
             "then device allows. " WORK_DIMENSIONS_STR,
             WORK_DIMENSIONS_VAL);
  } else if (!global_dims_divisible_by_local_dims) {
    is_ok = false;
    snprintf(msg_buffer, sizeof(msg_buffer),
             "Work parameters: For each dimension "
             "global_work_size should be multiply of "
             "local_work_size. " WORK_DIMENSIONS_STR,
             WORK_DIMENSIONS_VAL);
  } else if (real_global_work_size > device_work_id_range) {
    is_ok = false;
    snprintf(msg_buffer, sizeof(msg_buffer),
             "Work parameters: global_work_size(%llu) is bigger then device "
             "address_bits(%d) can represent. " WORK_DIMENSIONS_STR,
             real_global_work_size, device.address_bits, WORK_DIMENSIONS_VAL);
  } else if (real_local_work_size > device.max_work_group_size ||
             real_local_work_size > this->max_work_group_size) {
    is_ok = false;
    snprintf(msg_buffer, sizeof(msg_buffer),
             "Work parameters: local_work_size(%llu) is bigger then device(%d) "
             "or kernel(%d) allows. " WORK_DIMENSIONS_STR,
             real_local_work_size, device.max_work_group_size,
             this->max_work_group_size, WORK_DIMENSIONS_VAL);
  }

  context->check_error(is_ok, msg_buffer);
}

std::ostream &operator<<(std::ostream &os, opencl::Kernel &k) {
  os << "program id: " << k.program_id                                //
     << ", kernel id: " << k.kernel_id                                //
     << ", max_work_group_size: " << k.max_work_group_size            //
     << ", private_mem_size: " << k.private_mem_size                  //
     << ", pref_work_group_multiple: " << k.pref_work_group_multiple  //
     << ", allocated local memory: " << (k.current_local_memory() / 1024)
     << "KB";  //
  return os;
}
}
