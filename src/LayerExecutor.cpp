#include "LayerExecutor.hpp"

#include <algorithm>  // for std::min
#include <stdexcept>

#include "LayerData.hpp"
#include "opencl/Context.hpp"
#include "opencl/UtilsOpenCL.hpp"

namespace cnn_sr {

cl_event LayerExecutor::operator()(opencl::Kernel& kernel,
                                   const LayerData& data,
                                   opencl::MemoryHandler*& gpu_buf_in,
                                   size_t input_w, size_t input_h,
                                   opencl::MemoryHandler*& gpu_buf_out,
                                   cl_event* ev_to_wait_for) {
  opencl::Context* const context = kernel.get_context();
  size_t out_size[2];
  data.get_output_dimensions(out_size, input_w, input_h);
  size_t out_count = out_size[0] * out_size[1] * data.current_filter_count;
  std::cout << "out size: " << out_size[0] << "x" << out_size[1] << "x"
            << data.current_filter_count << "=" << out_count << std::endl;

  pre_exec_validation(data, gpu_buf_in, input_w, input_h);

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size, input_w,
                            input_h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;

  // TODO IMPORTANT: reuse gpu buffers for better performance

  // buffers: in_source, W, B , out_target
  /* clang-format off */
  auto gpu_buf_W = context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * data.weights.size());
  context->write_buffer(gpu_buf_W, (void *)data.weights_ptr(), true);
  auto gpu_buf_B = context->allocate( CL_MEM_READ_ONLY, sizeof(cl_float) * data.bias.size());
  context->write_buffer(gpu_buf_B, (void *)data.bias_ptr(), true);

  gpu_buf_out = context->allocate(CL_MEM_READ_WRITE, sizeof(cl_float) * out_count);
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
                                        opencl::MemoryHandler* input,
                                        size_t input_w, size_t input_h) {
  LayerData::validate(data);

  size_t input_size =  // TODO move all calcs. like this to LayerData
      input_w * input_h * data.n_prev_filter_cnt * sizeof(cl_float);

  if (input->size < input_size) {
    char buf[255];
    snprintf(buf, 255,
             "Declared input_w(%d)*input_h(%d)*n_prev_filter_cnt(%d)=%d "
             "is bigger then input array (%d elements)."
             " Expected more elements in input array.",
             input_w, input_h, data.n_prev_filter_cnt, input_size, input->size);
    throw std::runtime_error(buf);
  }
}

opencl::Kernel* LayerExecutor::create_layer_kernel(
    opencl::Context* const ctx, const char* const kernel_file,
    size_t current_filter_count, int result_multiply) {
  char buf[255];
  if (result_multiply) {
    snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=1 -D RESULT_MULTIPLY=%d",
             result_multiply);
    std::cout << "RESULT_MULTIPLY=" << result_multiply << " (last layer)"
              << std::endl;
  } else {
    // TODO current_filter_count=64 causes errors:
    // CL_INVALID_COMMAND_QUEUE (maybe memory alloc?)
    snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=%d", current_filter_count);
  }

  return ctx->create_kernel(kernel_file, buf);
}

//
}
