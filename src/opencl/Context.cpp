#include "Context.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "UtilsOpenCL.hpp"


namespace opencl {

//
// MemoryHandler
//

MemoryHandler::MemoryHandler()
    : handle(nullptr),
      released(false){
}

void MemoryHandler::release(){
  if(!released && handle)
    clReleaseMemObject(handle);
  released = true;
}

//
// Context
//

// init/core functions

Context::Context(int argc, char **argv)
    : initialized(false),
      argc(argc),
      argv(argv),
      _kernel_count(0),
      _allocation_count(0){
}

Context::~Context() {
  this->_cleanup();
}

void Context::init() {
  // TODO ad ability to select platform & device
  cl_int ciErr1;

  // Get an OpenCL platform
  cl_platform_id platform_id;
  ciErr1 = clGetPlatformIDs(1, &platform_id, nullptr);
  check_error(ciErr1, "Error in clGetPlatformID");
  platform_info(platform_id, this->_platform);
  std::cout << "PLATFORM: " << _platform << std::endl;

  // Get the devices
  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
             1, &_cldevice, nullptr);
  check_error(ciErr1, "Error in clGetDeviceIDs");
  device_info(_cldevice, this->_device);
  std::cout << "DEVICE:" << _device << std::endl;

  // Create the context
  _clcontext = clCreateContext(0, 1, &_cldevice, nullptr, nullptr, &ciErr1);
  check_error(ciErr1, "Error in clCreateContext");

  // Create a command-queue
  _clcommand_queue = clCreateCommandQueue(_clcontext, _cldevice, 0, &ciErr1);
  check_error(ciErr1, "Error in clCreateCommandQueue");

  initialized = true;
}

void Context::_cleanup(){
  // kernels
  for(size_t i = 0; i < _kernel_count; i++){
    _kernels[i].cleanup();
  }

  // memory
  for(size_t i = 0; i < _allocation_count; i++){
    MemoryHandler* k = _allocations + i;
    k->release();
  }

  // other
  if (_clcommand_queue)
    clReleaseCommandQueue(_clcommand_queue);
  if (_clcontext)
    clReleaseContext(_clcontext);
}

void Context::check_error(cl_int errCode, char const *msg) {
  // std::cout << "CHECK: " << errCode << ": " << msg << std::endl;
  if (errCode != CL_SUCCESS) {
    std::cout << msg
              << "[" << errCode << "]: "
              << utils::get_opencl_error_str(errCode) << std::endl;
    this->_cleanup();
    throw std::runtime_error(msg);
  }
}

void Context::check_error(bool check, char  const *msg){
  this->check_error(check? CL_SUCCESS : -100, msg);
}


// execution

MemoryHandler* Context::allocate(cl_mem_flags flags, size_t size){
  check_error(initialized, "Context was not initialized");
  check_error(_allocation_count < MAX_ALLOCATIONS_COUNT,
    "Wrapper hit allocations limit, increase MAX_ALLOCATIONS_COUNT");

  cl_int ciErr1;
  MemoryHandler* k = _allocations + _allocation_count;
  ++_allocation_count;
  k->handle = clCreateBuffer(_clcontext, flags, size, nullptr, &ciErr1);
  k->size = size;
  check_error(ciErr1, "Error in clCreateBuffer");
  return k;
}

Kernel* Context::create_kernel(char const *file_path,
                               char const *cmp_opt, char const *main_f){
  check_error(initialized, "Context was not initialized");
  check_error(_kernel_count < MAX_KERNEL_COUNT,
    "Wrapper hit kernel limit, increase MAX_KERNEL_COUNT");
  std::cout << "Reading kernel function from '" << file_path << "'" << std::endl;
  cl_int ciErr1;

  Kernel* k = _kernels + _kernel_count;
  ++_kernel_count;

  // TODO better manage the resources: kernel_source, program_id, kernel_id
  // (if code crashes there is going to be a leak !)

  // Read the OpenCL kernel from source file
  size_t kernel_len = 0;
  char* kernel_source = utils::load_file(file_path, "", &kernel_len);
  std::cout << "Kernel length: " << kernel_len << std::endl;
  check_error(kernel_len > 0, "Could not read file");

  // create program
  cl_program program_id = clCreateProgramWithSource(_clcontext,
                    1, (const char **)&kernel_source, &kernel_len, &ciErr1);
  check_error(ciErr1, "Error in clCreateProgramWithSource");
  free(kernel_source);

  // build program
  ciErr1 = clBuildProgram(program_id, 1, &_cldevice,
                          cmp_opt, nullptr, nullptr);
  if (ciErr1 == CL_BUILD_PROGRAM_FAILURE) {
    char buffer[2048];
    clGetProgramBuildInfo(program_id, _cldevice, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, nullptr);
    std::cout << "******************************************" << std::endl
              << "            --- Build log ---" << std::endl << std::endl
              << buffer << std::endl
              << "******************************************" << std::endl;
  }
  check_error(ciErr1, "Error in clBuildProgram");

  // Create the kernel
  cl_kernel kernel_id = clCreateKernel(program_id, main_f, &ciErr1);
  check_error(ciErr1, "Error in clCreateKernel");
  size_t max_work_group_size;
  ciErr1 = clGetKernelWorkGroupInfo(kernel_id, _cldevice,
     CL_KERNEL_WORK_GROUP_SIZE, 1024, &max_work_group_size, nullptr);

  k->init(this, kernel_id, program_id, max_work_group_size);

  // std::cout << "kernel created(f) :" <<k->kernel_id<<":"<<k->program_id<< std::endl;
  return k;
}

cl_event Context::read_buffer(MemoryHandler* gpu_buffer,
                              size_t offset, size_t size, void *dst,
                              bool block,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count){
  check_error(initialized, "Context was not initialized");
  check_error(size <= gpu_buffer->size, "Tried to read more then is allocated");
  cl_event finish_token;
  cl_bool clblock = block? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_buffer->handle;
  cl_int ciErr1 = clEnqueueReadBuffer(
    _clcommand_queue, gpu_memory_pointer, // what and where to execute
    clblock,           // block or not
    offset, size, dst, // read params: offset, size and target
    events_to_wait_for_count, events_to_wait_for, &finish_token); // sync events
  check_error(ciErr1, "Error in read buffer");
  return finish_token;
}

cl_event Context::read_buffer(MemoryHandler* gpu_buffer, void *dst, bool block,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count){
  return this->read_buffer(gpu_buffer, 0, gpu_buffer->size, dst, block,
                           events_to_wait_for, events_to_wait_for_count);
}

cl_event Context::write_buffer(MemoryHandler* gpu_buffer,
                               size_t offset, size_t size, void *src,
                               bool block,
                               cl_event* events_to_wait_for,
                               int events_to_wait_for_count){
  check_error(initialized, "Context was not initialized");
  check_error(size <= gpu_buffer->size, "Tried to write more then is allocated");
  cl_event finish_token;
  cl_bool clblock = block? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_buffer->handle;
  cl_int ciErr1 = clEnqueueWriteBuffer(
    _clcommand_queue, gpu_memory_pointer, // what and where to execute
    clblock,           // block or not
    offset, size, src, // read params: offset, size and target
    events_to_wait_for_count, events_to_wait_for, &finish_token); // sync events
  check_error(ciErr1, "Error in write buffer");
  return finish_token;
}

cl_event Context::write_buffer(MemoryHandler* gpu_buffer, void *src, bool block,
                               cl_event* events_to_wait_for,
                               int events_to_wait_for_count){
  return this->write_buffer(gpu_buffer, 0, gpu_buffer->size, src, block,
                            events_to_wait_for, events_to_wait_for_count);
}

cl_event Context::zeros_float(MemoryHandler* handler, bool block,
                              cl_event* es, int event_count){
  size_t len = handler->size / sizeof(float);
  std::vector<float> v;
  v.reserve(len);
  for (size_t i = 0; i < len; i++) {
    v.push_back(0.0f);
  }
  return this->write_buffer(handler, &v[0], block, es, event_count);
}

MemoryHandler* Context::create_image(cl_mem_flags flags,
                                     size_t w, size_t h,
                                     const cl_image_format *image_format){
  check_error(initialized, "Context was not initialized");
  check_error(_allocation_count < MAX_ALLOCATIONS_COUNT,
    "Wrapper hit allocations limit, increase MAX_ALLOCATIONS_COUNT");

  cl_int ciErr1;
  MemoryHandler* k = _allocations + _allocation_count;
  ++_allocation_count;
  k->handle = clCreateImage2D(_clcontext, flags,
                              image_format, w, h,
                              0, nullptr, &ciErr1);
  check_error(ciErr1, "Error in clCreateImage2D");
  return k;
}

cl_event Context::write_image(MemoryHandler* gpu_image,
                              utils::ImageData& data,
                              bool block,
                              cl_event* events_to_wait_for,
                              int events_to_wait_for_count){
  check_error(initialized, "Context was not initialized");
  cl_event finish_token;
  cl_bool clblock = block? CL_TRUE : CL_FALSE;
  cl_mem gpu_memory_pointer = gpu_image->handle;
  size_t origin[3] = {0,0,0};
  size_t region[3] = {(size_t)data.w, (size_t)data.h, 1};
  cl_int ciErr1 = clEnqueueWriteImage(
    _clcommand_queue, gpu_memory_pointer, // what and where to execute
    clblock,           // block or not
    origin, region,    //
    // sizeof(cl_uchar) * data.w * data.bpp, 0,
    sizeof(cl_uchar) * data.w * 4, 0,
    (void*)data.data,
    events_to_wait_for_count, events_to_wait_for, &finish_token); // sync events
  check_error(ciErr1, "Error in write_image");
  return finish_token;
}

// info

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
  // get base info
  ciErr1 = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
    sizeof(platform_info.name), &platform_info.name, &value_size);
  platform_info.name[value_size] = '\0';
  ciErr1 |= clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
    sizeof(platform_info.vendor), &platform_info.vendor, &value_size);
  platform_info.vendor[value_size] = '\0';
  ciErr1 |= clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,
    sizeof(platform_info.version), &platform_info.version, &value_size);
  platform_info.version[value_size] = '\0';
  check_error(ciErr1, "Could not get platform details");

  if(!devices){
    // no reason to read device data
    return;
  }

  // get device count
  cl_uint device_count = 0;
  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,
    0, nullptr, &device_count);
  check_error(ciErr1, "Could not get platform devices");
  // std::cout << "  found " << device_count << " devices" << std::endl;

  // device ids
  std::vector<cl_device_id> device_ids;
  device_ids.reserve(device_count);
  for (size_t i = 0; i < device_count; i++) {
    device_ids.push_back(0);
  }

  ciErr1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,
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
  ciErr1 |= clGetDeviceInfo(device_id, CL_DEVICE_NAME,
                            sizeof(info.name), &info.name, &value_size);
  info.name[value_size] = '\0';
  check_error(ciErr1, "Could not get device data");
}

}

std::ostream& operator<< (std::ostream& os, const opencl::PlatformInfo& platform_info){
  os << platform_info.vendor
     << "::" << platform_info.name
     << ", version " << platform_info.version;
  return os;
}

std::ostream& operator<< (std::ostream& os, const opencl::DeviceInfo& device){
  auto wifd = device.work_items_for_dims;
  os <<  opencl::utils::device_type_str[device.type]
     << "::" << device.name
     << ", memory: " << (device.global_mem_size / 1024 / 1024) << "MB"
     << ", address bits: " << device.address_bits
     << ", max work group size: " << device.max_work_group_size
     << ", work items: [" << wifd[0] << ", " << wifd[1] << ", " << wifd[2]
     << "], image support: " << (device.image_support==CL_TRUE ? "YES" : "NO");
  return os;
}

/*
int main(int argc, char **argv) {
  opencl::Context context(argc, argv);
  context.display_opencl_info();
}
*/
