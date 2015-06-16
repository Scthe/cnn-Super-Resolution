#include "LayerExecutor.hpp"

#include <algorithm>  // for std::min
#include <stdexcept>

#include "LayerData.hpp"
#include "opencl/Context.hpp"

size_t closest_power_of_2(int x);

using namespace opencl;

namespace cnn_sr {

cl_event LayerExecutor::operator()(opencl::Kernel& kernel,
                                   const LayerData& data,
                                   std::vector<float>& input,
                                   opencl::MemoryHandler*& gpu_buf_out,
                                   size_t input_w, size_t input_h,
                                   cl_event* ev_to_wait_for) {
  Context* const context = kernel.get_context();
  size_t out_size[2];
  data.get_output_dimensions(out_size, input_w, input_h);
  size_t out_count = out_size[0] * out_size[1] * data.current_filter_count,
         input_size = input_w * input_h * data.n_prev_filter_cnt;
  // std::cout << "out size: " << out_size[0] << "x" << out_size[1] <<
  // std::endl;

  pre_exec_validation(data, input, input_w, input_h);

  size_t global_work_size[2];
  size_t local_work_size[2];
  this->work_sizes(kernel, global_work_size, local_work_size, input_w, input_h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;

  // TODO IMPORTANT: reuse gpu buffers for better performance

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_in = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * input_size);
  context->write_buffer(gpu_buf_in, (void *)&input[0], true);
  auto gpu_buf_W = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data.weights.size());
  context->write_buffer(gpu_buf_W, (void *)data.weights_ptr(), true);
  auto gpu_buf_B = context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * data.bias.size());
  context->write_buffer(gpu_buf_B, (void *)data.bias_ptr(), true);

  gpu_buf_out = context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * out_count);
  context->zeros_float(gpu_buf_out, true);
  /* clang-format on */

  // args
  kernel.push_arg(gpu_buf_in);
  kernel.push_arg(gpu_buf_out);
  kernel.push_arg(gpu_buf_W);
  kernel.push_arg(gpu_buf_B);
  kernel.push_arg(sizeof(cl_uint), (void*)&data.n_prev_filter_cnt);
  kernel.push_arg(sizeof(cl_uint), (void*)&data.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void*)&input_w);
  kernel.push_arg(sizeof(cl_uint), (void*)&input_h);

  // run
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  return kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

void LayerExecutor::pre_exec_validation(const LayerData& data,
                                        std::vector<float>& input,
                                        size_t input_w, size_t input_h) {
  LayerData::validate(data);

  size_t input_size = input_w * input_h * data.n_prev_filter_cnt;
  if (input.size() < input_size) {
    char buf[255];
    snprintf(buf, 255,
             "Declared input_w(%d)*input_h(%d)*n_prev_filter_cnt(%d)=%d "
             "is bigger then input array (%d elements)."
             " Expected more elements in input array.",
             input_w, input_h, data.n_prev_filter_cnt, input_size,
             input.size());
    throw std::runtime_error(buf);
  }
}

void LayerExecutor::work_sizes(const opencl::Kernel& kernel,
                               size_t* global_work_size,
                               size_t* local_work_size, size_t w, size_t h) {
  auto context = kernel.get_context();
  auto device = context->device();
  auto max_local =
      std::min(device.max_work_group_size, kernel.get_max_work_group_size());
  auto max_device_local_size = device.work_items_for_dims;

  // size_t max_local = 1024; // TESTS ONLY
  // size_t max_device_local_size[2] = {128, 128}; // TESTS ONLY

  global_work_size[0] = closest_power_of_2(static_cast<int>(w));
  global_work_size[1] = closest_power_of_2(static_cast<int>(h));

  // if picture dimension is smaller then max allowed local group dimension
  // then just make it power of 2. Else we will use max value possible.
  local_work_size[0] = w >= max_device_local_size[0] ? max_device_local_size[0]
                                                     : closest_power_of_2(w);
  local_work_size[1] = h >= max_device_local_size[1] ? max_device_local_size[1]
                                                     : closest_power_of_2(h);

  bool div_w = local_work_size[0] > local_work_size[1];
  while (local_work_size[0] * local_work_size[1] > max_local) {
    // note: we are decreasing only one of local_work_size for each iteration
    local_work_size[0] /= div_w ? 2 : 1;
    local_work_size[1] /= !div_w ? 2 : 1;
    div_w = !div_w;
  }

  if (global_work_size[0] < local_work_size[0] ||
      global_work_size[1] < local_work_size[1] || local_work_size[0] == 0 ||
      local_work_size[1] == 0) {
    throw std::runtime_error("Tried to create nonstandard work dimensions");
  }
}

//
}

size_t closest_power_of_2(int x) {
  if (x < 0) return 0;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return static_cast<size_t>(x + 1);
}
