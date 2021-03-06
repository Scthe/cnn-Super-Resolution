#ifndef OPENCL_CONTEXT_H_
#define OPENCL_CONTEXT_H_

#include <vector>
#include <iostream>  // for std::ostream& operator<<(..)
#include "CL/opencl.h"
#include "Kernel.hpp"

#define MAX_INFO_STRING_LEN 256

namespace opencl {

class Context;

namespace utils {
struct ImageData;
}

/**
 * base information about platform
 */
struct PlatformInfo {
  char name[MAX_INFO_STRING_LEN];
  char vendor[MAX_INFO_STRING_LEN];
  char version[MAX_INFO_STRING_LEN];
};

/**
 * base information about device
 */
struct DeviceInfo {
  cl_device_id device_id;
  cl_device_type type;
  char name[MAX_INFO_STRING_LEN];
  cl_uint compute_units;
  cl_ulong global_mem_size;
  cl_ulong local_mem_size;
  cl_device_local_mem_type local_mem_type;
  cl_uint address_bits;
  size_t max_work_group_size;
  size_t work_items_for_dims[3];
  cl_bool image_support;
};

/**
 * opencl memory handle
 */
typedef size_t MemoryHandle;

/**
 * represents gpu memory allocation. Should not be used
 */
struct RawMemoryHandle {
  RawMemoryHandle();
  void release();
  inline bool is_usable() const { return !released; }
  inline bool is_image() const { return bpp != 0; }

  cl_mem handle;
  size_t size = 0;
  /* must be nonzero if represents image */
  size_t bpp = 0;

 private:
  bool released;
};

/**
 * Base class for interaction with opencl.
 * Remember to call init!
 */
class Context {
 public:
  Context();
  ~Context();
  void init(bool profile = false);
  void check_error(bool, char const*);
  void check_error(cl_int, char const*);
  void print_app_memory_usage();

  //
  // execution
  //

  void block();

  /**
   * Allocate memory on opencl device
   * https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateBuffer.html
   *
   * @param  flags    opencl flags
   * @param  size     bytest to allocate. Use f.e. sizeof(cl_char) * COUNT
   * @return          handler used by context
   */
  MemoryHandle allocate(cl_mem_flags, size_t);

  /**
   * Create kernel from file
   *
   * @param  file_path path to .cl file that contains source code
   * @param  cmp_opt   [OPT] compilation options f.e. macros
   * @param  main_f    [OPT] name of main kernel function
   * @return           kernel object
   */
  Kernel* create_kernel(char const* file_path, char const* cmp_opt = nullptr,
                        char const* main_f = "main");

  /**
   * Read buffer from opencl device and copy it to host memory
   *
   * @param  gpu_buffer               source buffer
   * @param  offset                   buffer offset
   * @param  size                     how much to read
   * @param  dst                      destination buffer
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event read_buffer(MemoryHandle, size_t offset, size_t size, void* dst,
                       bool block, cl_event* es = nullptr, int event_count = 0);

  /**
   * Read buffer from opencl device and copy it to host memory
   *
   * @param  gpu_buffer               source buffer
   * @param  dst                      destination buffer
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event read_buffer(MemoryHandle, void* dst, bool block,
                       cl_event* es = nullptr, int event_count = 0);

  /* clang-format off */
 /**
  * Copy data from host memory to opencl device
  *
  * @param  gpu_buffer               destination buffer
  * @param  offset                   buffer offset
  * @param  size                     how much to read
  * @param  src                      source buffer
  * @param  block                    blocking/nonblocking operation switch
  * @param  events_to_wait_for       [OPT]wait for other operations to finish
  * @param  events_to_wait_for_count [OPT]
  * @return                          opencl event object
  */
  cl_event write_buffer(MemoryHandle, size_t offset, size_t size, void* src,
                        bool block, cl_event* es = nullptr,
                        int event_count = 0);
  /* clang-format on */

  /**
   * Copy data from host memory to opencl device
   *
   * @param  gpu_buffer               destination buffer
   * @param  src                      source buffer
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event write_buffer(MemoryHandle, void* src, bool block,
                        cl_event* es = nullptr, int event_count = 0);

  /**
   * Fill with zero values
   *
   * @param  gpu_buffer               destination buffer
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event zeros_float(MemoryHandle, bool block,  //
                       cl_event* es = nullptr, int event_count = 0);

  /**
   * Fill buffer
   *
   * @param  gpu_buffer               destination buffer
   * @param  value                    value to fill with
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event fill_float(MemoryHandle, float, bool block,  //
                      cl_event* es = nullptr, int event_count = 0);

  /**
   * Copy from source to destination
   * @param  src_buffer               source
   * @param  dst_buffer               destination
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event copy_buffer(MemoryHandle, MemoryHandle,  //
                       cl_event* es = nullptr, int event_count = 0);

  /**
   * Copy from source to destination
   * @param  src_buffer               source
   * @param  dst_buffer               destination
   * @param  dst_offset               destination write offset
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event copy_buffer(MemoryHandle, MemoryHandle, size_t,  //
                       cl_event* es = nullptr, int event_count = 0);

  /**
 * Allocate image
   *
   * @param  flags        opencl flags
   * @param  w            width
   * @param  h            height
   * @param  image_format
   * @return              handler used by context
   */
  MemoryHandle create_image(cl_mem_flags, cl_channel_order, cl_channel_type,
                            size_t, size_t);

  /**
   * Write image data to buffer
   *
   * @param  gpu_image                memory handler to write to
   * @param  data                     dara to write
   * @param  block                    blocking/nonblocking operation switch
   * @param  events_to_wait_for       [OPT]wait for other operations to finish
   * @param  events_to_wait_for_count [OPT]
   * @return                          opencl event object
   */
  cl_event write_image(MemoryHandle, utils::ImageData&, bool block,
                       cl_event* es = nullptr, int event_count = 0);

  //
  // info
  //

  /**
   * print to stdout platform and devices information
   */
  void display_opencl_info();

  //
  // get&set
  //

  /**
   * was context initialized ? AKA can I use any cmds beside info gathering ?
   */
  bool is_initialized() { return initialized; }

  /**
   * device that this context is bound to
   */
  DeviceInfo device() { return _device; }

  /**
   * platform that this context is bound to
   */
  PlatformInfo platform() { return _platform; }

  /** code profile mode - kernel execution timings etc. */
  bool is_running_profile_mode() { return _profiling; }

  /**
   * command queue. This may be called leaky abstraction, but it's not like
   * we don't expose more advanced stuff (f.e. max_work_group_size).
   * Also You probably will not have any use of raw command_queue.
   */
  cl_command_queue* command_queue() { return &_clcommand_queue; }

  RawMemoryHandle* raw_memory(MemoryHandle);

 private:
  void _cleanup();
  size_t channels_count(cl_channel_order, cl_channel_type);
  size_t per_pixel_bytes(cl_channel_order, cl_channel_type);
  void platform_info(cl_platform_id platform_id, PlatformInfo& platform_info,
                     std::vector<DeviceInfo>* devices = nullptr);
  void device_info(cl_device_id, DeviceInfo&);

 private:
  bool initialized;
  cl_context _clcontext;
  cl_command_queue _clcommand_queue;
  bool _profiling;

  DeviceInfo _device;
  PlatformInfo _platform;

  std::vector<Kernel> _kernels;
  std::vector<RawMemoryHandle> _allocations;
};
}

std::ostream& operator<<(std::ostream&, const opencl::PlatformInfo&);
std::ostream& operator<<(std::ostream&, const opencl::DeviceInfo&);

#endif  // OPENCL_CONTEXT_H_
