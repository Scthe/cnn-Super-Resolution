#include "Context.hpp"

#include <iostream>
#include <stdexcept>

#include "UtilsOpenCL.hpp"
#include "../pch.hpp"

bool print_info = false;

/**
 * _kernels uses pointers, which makes the wrapper more lightweight.
 * As soon as vector that holds original instances is reloacted
 * the pointers are obsolete.
 */
const size_t max_resources_per_type = 128;

namespace opencl {

//
// RawMemoryHandle
//

RawMemoryHandle::RawMemoryHandle() : handle(nullptr), released(false) {}

void RawMemoryHandle::release() {
  if (!released && handle) {
    clReleaseMemObject(handle);
    // auto ciErr1 = clReleaseMemObject(handle); // TODO check error
    // check_error(ciErr1, "Error in RawMemoryHandle::release");
  }
  released = true;
}

//
// Context
//

// init/core functions

Context::Context() : initialized(false) {}

Context::~Context() { this->_cleanup(); }

void Context::init(bool profile) {
  // TODO ad ability to select platform & device
  cl_int ciErr1;
  _profiling = profile;

  // Get an OpenCL platform
  cl_platform_id platform_id;
  ciErr1 = clGetPlatformIDs(1, &platform_id, nullptr);
  check_error(ciErr1, "Error in clGetPlatformID");
  platform_info(platform_id, this->_platform);
  std::cout << "PLATFORM: " << _platform << std::endl;

  // Get the devices
  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                          &_device.device_id, nullptr);
  check_error(ciErr1, "Error in clGetDeviceIDs");
  device_info(_device.device_id, this->_device);
  std::cout << "DEVICE:" << _device << std::endl;

  // Create the context
  _clcontext =
      clCreateContext(0, 1, &_device.device_id, nullptr, nullptr, &ciErr1);
  check_error(ciErr1, "Error in clCreateContext");

  // Create a command-queue
  _clcommand_queue =
      clCreateCommandQueue(_clcontext, _device.device_id,
                           _profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &ciErr1);
  check_error(ciErr1, "Error in clCreateCommandQueue");

  _kernels.reserve(max_resources_per_type);
  _allocations.reserve(max_resources_per_type);

  initialized = true;
}

void Context::_cleanup() {
  // can be called only once
  if (!initialized) return;
  initialized = false;
  this->block();

  // kernels
  for (auto kernel = begin(_kernels); kernel != end(_kernels); ++kernel) {
    // we are done with the kernel, print it's execution time sum
    if (_profiling) {
      auto timer = kernel->get_total_execution_time();
      auto timer_in_s = timer / 1000000000.0;
      std::cout << "Kernel " << kernel->get_human_identifier()
                << " total execution time: " << timer << "ns = " << timer_in_s
                << "s" << std::endl;
    }

    kernel->cleanup();
  }

  // memory
  for (auto alloc = begin(_allocations); alloc != end(_allocations); ++alloc) {
    alloc->release();
  }

  // other
  if (_clcommand_queue) clReleaseCommandQueue(_clcommand_queue);
  if (_clcontext) clReleaseContext(_clcontext);
}

void Context::check_error(cl_int errCode, char const* msg) {
  // std::cout << "CHECK: " << errCode << ": " << msg << std::endl;
  if (errCode != CL_SUCCESS) {
    std::cout << "[OPENCL ERROR] " << utils::get_opencl_error_str(errCode)
              << "(" << errCode << ") : " << msg << std::endl;
    this->_cleanup();
    throw std::runtime_error(msg);
  }
}

void Context::check_error(bool check, char const* msg) {
  this->check_error(check ? CL_SUCCESS : -100, msg);
}

RawMemoryHandle* Context::raw_memory(MemoryHandle handle) {
  check_error(handle < _allocations.size(),
              "Invalid memory handle."
              "Could not get RawMemoryHandle object");
  return &_allocations[handle];
}

void Context::print_app_memory_usage() {
  size_t image_memory = 0, buffer_memory = 0;
  for (const RawMemoryHandle& mem : _allocations) {
    if (!mem.is_usable()) continue;
    if (mem.is_image()) {
      image_memory += mem.size;
    } else {
      buffer_memory += mem.size;
    }
  }
  const size_t unit = 1024 * 1024;
  std::cout << "Memory usage: " << (image_memory + buffer_memory) / unit  //
            << "/" << _device.global_mem_size / unit << " MB ("           //
            << (image_memory + buffer_memory) * 100.0 / _device.global_mem_size
            << "%), " << buffer_memory / unit  //
            << "MB of raw buffers and " << image_memory / unit
            << "MB for images" << std::endl;
}

// core: execution related

void Context::block() {
  if (cnn_sr::warn_about_blocking_operation)
    std::cout << "BLOCK explicit Context::block()" << std::endl;
  cl_int ciErr1;
  ciErr1 = clFlush(_clcommand_queue);
  check_error(ciErr1,
              "Error during command queue flush during Context::block()");
  ciErr1 = clFinish(_clcommand_queue);
  check_error(ciErr1, "Error during clFinish during Context::block()");
}

MemoryHandle Context::allocate(cl_mem_flags flags, size_t size) {
  check_error(initialized, "Context was not initialized");

  cl_int ciErr1;
  _allocations.push_back(RawMemoryHandle());
  MemoryHandle idx = _allocations.size() - 1;
  auto mem_handle = &_allocations[idx];
  mem_handle->handle =
      clCreateBuffer(_clcontext, flags, size, nullptr, &ciErr1);
  mem_handle->size = size;
  check_error(ciErr1, "Error in clCreateBuffer");
  return idx;
}

Kernel* Context::create_kernel(char const* file_path, char const* cmp_opt,
                               char const* main_f) {
  check_error(initialized, "Context was not initialized");
  check_error(
      _kernels.size() < max_resources_per_type,
      "Kernel limit reached, increase max_resources_per_type in Context.cpp");
  if (print_info)
    std::cout << "Reading kernel function from '" << file_path
              << "' with args: '" << (cmp_opt ? cmp_opt : "") << "'"
              << std::endl;
  cl_int ciErr1;

  _kernels.push_back(Kernel());
  auto kernel_ptr = &_kernels[_kernels.size() - 1];

  // TODO better manage the resources: kernel_source, program_id, kernel_id
  // (if code crashes there is going to be a leak!)

  // Read the OpenCL kernel from source file
  size_t kernel_len = 0;
  char* kernel_source = utils::load_file(file_path, "", &kernel_len);
  if (print_info) std::cout << "Kernel length: " << kernel_len << std::endl;
  check_error(kernel_len > 0, "Could not read file");

  // create program
  cl_program program_id = clCreateProgramWithSource(
      _clcontext, 1, (const char**)&kernel_source, &kernel_len, &ciErr1);
  check_error(ciErr1, "Error in clCreateProgramWithSource");
  free(kernel_source);

  // build program
  ciErr1 = clBuildProgram(program_id, 1, &_device.device_id, cmp_opt, nullptr,
                          nullptr);
  if (ciErr1 == CL_BUILD_PROGRAM_FAILURE) {
    char buffer[2048];
    clGetProgramBuildInfo(program_id, _device.device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, nullptr);
    std::cout << "******************************************" << std::endl
              << "            --- Build log ---" << std::endl
              << std::endl
              << buffer << std::endl
              << "******************************************" << std::endl;
  }
  check_error(ciErr1, "Error in clBuildProgram");

  // Create the kernel
  cl_kernel kernel_id = clCreateKernel(program_id, main_f, &ciErr1);
  check_error(ciErr1, "Error in clCreateKernel");

  kernel_ptr->init(this, kernel_id, program_id, file_path, cmp_opt);

  return kernel_ptr;
}

///
/// Buffers: read/write/copy
///
cl_event Context::read_buffer(MemoryHandle gpu_buffer_handle, size_t offset,
                              size_t size, void* dst, bool block,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count) {
  if (cnn_sr::warn_about_blocking_operation && block)
    std::cout << "BLOCK: read_buffer" << std::endl;
  check_error(initialized, "Context was not initialized");
  auto gpu_buffer = raw_memory(gpu_buffer_handle);
  check_error(size <= gpu_buffer->size, "Tried to read more then is allocated");
  cl_event finish_token;
  cl_bool clblock = block ? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_buffer->handle;
  cl_int ciErr1 = clEnqueueReadBuffer(
      _clcommand_queue, gpu_memory_pointer,  // what and where to execute
      clblock,                               // block or not
      offset, size, dst,  // read params: offset, size and target
      events_to_wait_for_count, events_to_wait_for,  // sync events
      &finish_token);
  check_error(ciErr1, "Error in read buffer");
  return finish_token;
}

cl_event Context::read_buffer(MemoryHandle gpu_buffer_handle, void* dst,
                              bool block, cl_event* events_to_wait_for,
                              int events_to_wait_for_count) {
  auto gpu_buffer = raw_memory(gpu_buffer_handle);
  return this->read_buffer(gpu_buffer_handle, 0, gpu_buffer->size, dst, block,
                           events_to_wait_for, events_to_wait_for_count);
}

cl_event Context::write_buffer(MemoryHandle gpu_buffer_handle, size_t offset,
                               size_t size, void* src, bool block,
                               cl_event* events_to_wait_for,
                               int events_to_wait_for_count) {
  if (cnn_sr::warn_about_blocking_operation && block)
    std::cout << "BLOCK: write_buffer" << std::endl;
  check_error(initialized, "Context was not initialized");
  auto gpu_buffer = raw_memory(gpu_buffer_handle);
  check_error(size <= gpu_buffer->size,
              "Tried to write more then is allocated");
  cl_event finish_token;
  cl_bool clblock = block ? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_buffer->handle;
  cl_int ciErr1 = clEnqueueWriteBuffer(
      _clcommand_queue, gpu_memory_pointer,  // what and where to execute
      clblock,                               // block or not
      offset, size, src,  // read params: offset, size and target
      events_to_wait_for_count, events_to_wait_for,  // sync events
      &finish_token);
  check_error(ciErr1, "Error in write buffer");
  return finish_token;
}

cl_event Context::write_buffer(MemoryHandle gpu_buffer_handle, void* src,
                               bool block, cl_event* events_to_wait_for,
                               int events_to_wait_for_count) {
  auto gpu_buffer = raw_memory(gpu_buffer_handle);
  return this->write_buffer(gpu_buffer_handle, 0, gpu_buffer->size, src, block,
                            events_to_wait_for, events_to_wait_for_count);
}

cl_event Context::zeros_float(MemoryHandle gpu_buffer_handle, bool block,
                              cl_event* es, int event_count) {
  return fill_float(gpu_buffer_handle, 0.0f, block, es, event_count);
}

cl_event Context::fill_float(MemoryHandle gpu_buffer_handle, float val,
                             bool block, cl_event* es, int event_count) {
  auto gpu_buffer = raw_memory(gpu_buffer_handle);
  size_t len = gpu_buffer->size / sizeof(float);
  std::vector<float> v(len);
  for (size_t i = 0; i < len; i++) {
    v[i] = val;
  }
  return this->write_buffer(gpu_buffer_handle, &v[0], block, es, event_count);
}

cl_event Context::copy_buffer(MemoryHandle src_buffer, MemoryHandle dst_buffer,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count) {
  auto gpu_src = raw_memory(src_buffer);
  auto gpu_dst = raw_memory(dst_buffer);
  check_error(
      gpu_src->size == gpu_dst->size,
      "When performing buffer copy, both buffers should have equal length");
  return copy_buffer(src_buffer, dst_buffer, 0, events_to_wait_for,
                     events_to_wait_for_count);
}

cl_event Context::copy_buffer(MemoryHandle src_buffer, MemoryHandle dst_buffer,
                              size_t dst_offset, cl_event* events_to_wait_for,
                              int events_to_wait_for_count) {
  check_error(initialized, "Context was not initialized");
  auto gpu_src = raw_memory(src_buffer);
  auto gpu_dst = raw_memory(dst_buffer);
  check_error(gpu_src->size + dst_offset <= gpu_dst->size,
              "When performing buffer copy, would write after dst end");
  cl_event finish_token;
  cl_int ciErr1 = clEnqueueCopyBuffer(_clcommand_queue,              //
                                      gpu_src->handle,               //
                                      gpu_dst->handle,               //
                                      0, dst_offset, gpu_src->size,  //
                                      events_to_wait_for_count,
                                      events_to_wait_for, &finish_token);
  check_error(ciErr1, "Error in copy buffer");
  return finish_token;
}

///
/// Images
///
size_t Context::channels_count(cl_channel_order o, cl_channel_type t) {
  // 'x' means not all bits will be used f.e. CL_UNORM_SHORT_555
  switch (o) {
    case CL_R:
    case CL_A:
    case CL_Rx:
      return 1;
    case CL_INTENSITY:
    case CL_LUMINANCE:
      check_error(t == CL_UNORM_INT8 || t == CL_UNORM_INT16 ||
                      t == CL_SNORM_INT8 || t == CL_SNORM_INT16 ||
                      t == CL_HALF_FLOAT || t == CL_FLOAT,
                  "Use CL_INTENSITY/CL_LUMINANCE only with CL_UNORM_INT8, "
                  "CL_UNORM_INT16, "
                  "CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT");
      return 1;
    case CL_RG:
    case CL_RA:
    case CL_RGx:
      return 2;
    case CL_RGB:
    case CL_RGBx:
      check_error(t == CL_UNORM_SHORT_565 || t == CL_UNORM_SHORT_555 ||
                      t == CL_UNORM_INT_101010,
                  "Use CL_RGB/CL_RGBx only with CL_UNORM_SHORT_565, "
                  "CL_UNORM_SHORT_555 or CL_UNORM_INT_101010");
      return 3;
    case CL_RGBA:
      return 4;
    case CL_BGRA:
    case CL_ARGB:
      check_error(t == CL_UNORM_INT8 || t == CL_SNORM_INT8 ||
                      t == CL_SIGNED_INT8 || t == CL_UNSIGNED_INT8,
                  "Use CL_ARGB/CL_BGRA only with CL_UNORM_INT8, CL_SNORM_INT8, "
                  "CL_SIGNED_INT8 or CL_UNSIGNED_INT8");
      return 4;
  }
  check_error(false, "Unrecognised cl_channel_order");
  return 0;
}

size_t Context::per_pixel_bytes(cl_channel_order o, cl_channel_type t) {
  auto ch_count = channels_count(o, t);
  switch (t) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      return ch_count;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
      return 2 * ch_count;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
      return 4 * ch_count;
    case CL_UNORM_SHORT_565:
      check_error(o == CL_RGB || o == CL_RGB,
                  "Use CL_UNORM_SHORT_565 only with CL_RGB/CL_RGBx");
      return sizeof(cl_short);
    case CL_UNORM_SHORT_555:
      check_error(o == CL_RGB || o == CL_RGB,
                  "Use CL_UNORM_SHORT_555 only with CL_RGB/CL_RGBx");
      return sizeof(cl_short);
    case CL_UNORM_INT_101010:
      check_error(o == CL_RGB || o == CL_RGB,
                  "Use CL_UNORM_INT_101010 only with CL_RGB/CL_RGBx");
      return sizeof(cl_int);
    case CL_HALF_FLOAT:
      return sizeof(cl_float) * ch_count / 2;
    case CL_FLOAT:
      return sizeof(cl_float) * ch_count;
  }
  check_error(false, "Unrecognised cl_channel_type");
  return 0;
}

MemoryHandle Context::create_image(cl_mem_flags flags,
                                   cl_channel_order image_channel_order,
                                   cl_channel_type image_channel_data_type,
                                   size_t w, size_t h) {
  check_error(initialized, "Context was not initialized");
  auto bpp = per_pixel_bytes(image_channel_order, image_channel_data_type);

  cl_image_format image_format;
  image_format.image_channel_order = image_channel_order;
  image_format.image_channel_data_type = image_channel_data_type;

  cl_int ciErr1;
  _allocations.push_back(RawMemoryHandle());
  auto mem_idx = _allocations.size() - 1;
  auto mem_handle = &_allocations[mem_idx];

  mem_handle->handle = clCreateImage2D(_clcontext, flags, &image_format, w, h,
                                       0, nullptr, &ciErr1);
  mem_handle->size = w * h * bpp;
  mem_handle->bpp = bpp;
  check_error(ciErr1, "Error in clCreateImage2D");
  return mem_idx;
}

cl_event Context::write_image(MemoryHandle gpu_buffer_handle,
                              utils::ImageData& data, bool block,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count) {
  check_error(initialized, "Context was not initialized");
  auto gpu_image = raw_memory(gpu_buffer_handle);
  cl_event finish_token;
  cl_bool clblock = block ? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_image->handle;
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {(size_t)data.w, (size_t)data.h, 1};
  cl_int ciErr1 = clEnqueueWriteImage(
      _clcommand_queue, gpu_memory_pointer,  // what and where to execute
      clblock,                               // block or not
      origin, region,                        // corners: left-top, right-down
      data.w * gpu_image->bpp,               // length of each row in bytes
      0, (void*)data.data,                   //
      events_to_wait_for_count, events_to_wait_for,  // sync events
      &finish_token);
  check_error(ciErr1, "Error in write_image");
  return finish_token;
}

///
/// info
///

void Context::display_opencl_info() {
  cl_int ciErr1;

  cl_uint platform_count = 0;
  ciErr1 = clGetPlatformIDs(0, nullptr, &platform_count);
  check_error(ciErr1, "Could not get platform count");
  std::cout << "platforms:" << std::endl;

  // prepare platform ids vector
  std::vector<cl_platform_id> platform_ids;
  platform_ids.reserve(platform_count);
  for (size_t i = 0; i < platform_count; i++) {
    platform_ids.push_back(nullptr);
  }

  ciErr1 = clGetPlatformIDs(platform_count, &platform_ids[0], nullptr);
  check_error(ciErr1, "Could not get platform ids");

  PlatformInfo platform_info;
  std::vector<DeviceInfo> devices;
  for (auto i = begin(platform_ids); i != end(platform_ids); ++i) {
    devices.clear();
    this->platform_info(*i, platform_info, &devices);
    std::cout << "  " << platform_info << std::endl;
    std::cout << "  devices:" << std::endl;
    // devices
    for (auto j = begin(devices); j != end(devices); ++j) {
      std::cout << "    " << (*j) << std::endl;
    }
  }

  std::cout << "found " << platform_count << " opencl platforms" << std::endl;
}

void Context::platform_info(cl_platform_id platform_id,
                            PlatformInfo& platform_info,
                            std::vector<DeviceInfo>* devices) {
  size_t value_size = 0;
  cl_int ciErr1;

  /* clang-format off */
  ciErr1 = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
    sizeof(platform_info.name), &platform_info.name, &value_size);
  platform_info.name[value_size] = '\0';
  ciErr1 |= clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
    sizeof(platform_info.vendor), &platform_info.vendor, &value_size);
  platform_info.vendor[value_size] = '\0';
  ciErr1 |= clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,
    sizeof(platform_info.version), &platform_info.version, &value_size);
  platform_info.version[value_size] = '\0';
  /* clang-format on */

  check_error(ciErr1, "Could not get platform details");

  if (!devices) {
    // no reason to read device data
    return;
  }

  // get device count
  cl_uint device_count = 0;
  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,  //
                          0, nullptr, &device_count);
  check_error(ciErr1, "Could not get platform devices");
  // std::cout << "  found " << device_count << " devices" << std::endl;

  // device ids
  std::vector<cl_device_id> device_ids;
  device_ids.reserve(device_count);
  for (size_t i = 0; i < device_count; i++) {
    device_ids.push_back(0);
  }

  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,  //
                          device_count, &device_ids[0], nullptr);
  check_error(ciErr1, "Could not get device ids");

  for (auto i = begin(device_ids); i != end(device_ids); ++i) {
    DeviceInfo d;
    this->device_info(*i, d);
    devices->push_back(d);
  }
}

void Context::device_info(cl_device_id device_id, DeviceInfo& info) {
  cl_int ciErr1;
  size_t value_size = 0;
  /* clang-format off */
  ciErr1 =  clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE,
                            1024, &info.global_mem_size, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT,
                            1024, &info.image_support, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            1024, &info.max_work_group_size, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS,
                            1024, &info.address_bits, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            1024, &info.work_items_for_dims, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_TYPE,
                            1024, &info.type, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE,
                            1024, &info.local_mem_size, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE,
                            1024, &info.local_mem_type, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                            1024, &info.compute_units, nullptr);
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_NAME,
                            sizeof(info.name), &info.name, &value_size);
  info.name[value_size] = '\0';
  /* clang-format on */

  check_error(ciErr1, "Could not get device data");
}
}

std::ostream& operator<<(std::ostream& os,
                         const opencl::PlatformInfo& platform_info) {
  os << platform_info.vendor << "::" << platform_info.name  //
     << ", version " << platform_info.version;
  return os;
}

std::ostream& operator<<(std::ostream& os, const opencl::DeviceInfo& device) {
  auto wifd = device.work_items_for_dims;
  /* clang-format off */
  os << opencl::utils::device_type_str[device.type]
     << "::" << device.name
     << ", compute units: " << device.compute_units
     << ", global memory: " << (device.global_mem_size / 1024 / 1024) << "MB"
     << ", max local memory: " << (device.local_mem_size / 1024) << "KB, local memory type: "
          << (device.local_mem_type == CL_LOCAL ? "local" : "global")
     << ", address bits: " << device.address_bits
     << ", max work group size: " << device.max_work_group_size
     << ", work items: [" << wifd[0] << ", " << wifd[1] << ", " << wifd[2]
     << "], image support: " << (device.image_support == CL_TRUE ? "YES" : "NO");
  /* clang-format on */
  return os;
}

/*
int main(int argc, char **argv) {
  opencl::Context context(argc, argv);
  context.display_opencl_info();
}
*/
