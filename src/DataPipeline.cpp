#include "DataPipeline.hpp"

#include <stdexcept>  // std::runtime_error
#include <cstdio>     // snprintf
#include <vector>     // to hold subresults during cpu step of backpropagation

#include "LayerData.hpp"
#include "opencl/Context.hpp"
#include "opencl/UtilsOpenCL.hpp"

const char *const luma_kernel_file = "src/kernel/extract_luma.cl";
const char *const layer_kernel_file = "src/kernel/layer_uber_kernel.cl";
const char *const mse_kernel_file = "src/kernel/mse.cl";
const char *const sum_kernel_file = "src/kernel/sum.cl";
const char *const deltas_kernel_file = "src/kernel/layer_deltas.cl";
const char *const backpropagate_kernel_file = "src/kernel/backpropagate.cl";
const char *const subtract_from_all_kernel_file =
    "src/kernel/subtract_from_all.cl";

namespace cnn_sr {

int DataPipeline::LOAD_KERNEL_LUMA = 1;
int DataPipeline::LOAD_KERNEL_LAYERS = 2;
int DataPipeline::LOAD_KERNEL_MISC = 4;
int DataPipeline::LOAD_KERNEL_BACKPROPAGATE = 8;
int DataPipeline::LOAD_KERNEL_NONE = 0;
int DataPipeline::LOAD_KERNEL_ALL = DataPipeline::LOAD_KERNEL_LUMA |  //
                                    DataPipeline::LOAD_KERNEL_LAYERS |
                                    DataPipeline::LOAD_KERNEL_BACKPROPAGATE |
                                    DataPipeline::LOAD_KERNEL_MISC;

///
/// Construction/init/misc
///

DataPipeline::DataPipeline(opencl::Context *context)
    : _context(context),
      _initialized(false),
      _luma_kernel_norm(nullptr),
      _luma_kernel_raw(nullptr),
      _mse_kernel(nullptr),
      _sum_kernel(nullptr),
      _subtract_from_all_kernel(nullptr) {}

void DataPipeline::init(int load_flags) {
  load_kernels(load_flags);
  _initialized = true;
}

void DataPipeline::check_initialized(int kernel_load_flags) {
  if (!_initialized) {
    throw std::runtime_error(
        "Tried to use DataPipeline before it was initialized");
  }

  this->load_kernels(kernel_load_flags);
}

bool DataPipeline::allocation_has_right_size(opencl::MemoryHandler *alloc,
                                             size_t size) {
  if (!alloc) return false;
  if (alloc->size == size) return true;

  std::cout << "Was forced to realocate gpu buffer. This is not optimal and "
               "may be a bug." << std::endl;
  alloc->release();
  return false;
}

///
/// Kernel loading
///

void DataPipeline::load_kernels(int load_flags) {
  bool load_luma = (load_flags & DataPipeline::LOAD_KERNEL_LUMA) != 0,
       load_misc = (load_flags & DataPipeline::LOAD_KERNEL_MISC) != 0;

  if (load_luma) {
    auto norm_arg = "-D NORMALIZE";
    if (!_luma_kernel_norm)
      _luma_kernel_norm = _context->create_kernel(luma_kernel_file, norm_arg);
    if (!_luma_kernel_raw)
      _luma_kernel_raw = _context->create_kernel(luma_kernel_file);
  }

  if (load_misc) {
    if (!_mse_kernel) _mse_kernel = _context->create_kernel(mse_kernel_file);
    if (!_sum_kernel) _sum_kernel = _context->create_kernel(sum_kernel_file);
    if (!_subtract_from_all_kernel)
      _subtract_from_all_kernel =
          _context->create_kernel(subtract_from_all_kernel_file);
  }
}

opencl::Kernel *DataPipeline::create_layer_kernel(const LayerData &d,
                                                  int result_multiply) {
  char buf[255];
  if (result_multiply) {
    snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=1 -D RESULT_MULTIPLY=%d",
             result_multiply);
    std::cout << "RESULT_MULTIPLY=" << result_multiply << " (last layer)"
              << std::endl;
  } else {
    // TODO current_filter_count=64 causes errors:
    // CL_INVALID_COMMAND_QUEUE (maybe gpu memory alloc?)
    snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=%d", d.current_filter_count);
  }

  return _context->create_kernel(layer_kernel_file, buf);
}

opencl::Kernel *DataPipeline::create_deltas_kernel(const LayerData &d) {
  char buf[255];
  snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=%d", d.n_prev_filter_cnt);
  return _context->create_kernel(deltas_kernel_file, buf);
}

opencl::Kernel *DataPipeline::create_backpropagation_kernel(
    const LayerData &d) {
  size_t per_filter_size =
      d.f_spatial_size * d.f_spatial_size * d.n_prev_filter_cnt;
  char buf[255];
  snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=%d -D PER_FILTER_SIZE=%d",
           d.current_filter_count, per_filter_size);
  return _context->create_kernel(backpropagate_kernel_file, buf);
}

///
/// execute: misc
///

cl_event DataPipeline::extract_luma(opencl::utils::ImageData &img_data,
                                    opencl::MemoryHandler *&gpu_buf_raw_img,
                                    opencl::MemoryHandler *&gpu_buf_luma,
                                    bool normalize, cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_LUMA);

  size_t out_pixel_count = img_data.w * img_data.h;
  auto kernel = normalize ? _luma_kernel_norm : _luma_kernel_raw;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*kernel, global_work_size, local_work_size,
                            img_data.w, img_data.h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;

  // memory allocation
  if (!allocation_has_right_size(gpu_buf_raw_img, out_pixel_count)) {
    gpu_buf_raw_img = _context->create_image(
        CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  }
  _context->write_image(gpu_buf_raw_img, img_data, true);
  if (!allocation_has_right_size(gpu_buf_luma, out_pixel_count)) {
    gpu_buf_luma = _context->allocate(CL_MEM_READ_WRITE,
                                      sizeof(cl_float) * out_pixel_count);
  }

  // kernel args
  kernel->push_arg(gpu_buf_raw_img);
  kernel->push_arg(gpu_buf_luma);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.h);

  // Launch kernel
  auto finish_token = kernel->execute(2, global_work_size, local_work_size,
                                      ev_to_wait_for, ev_to_wait_for ? 1 : 0);
  return finish_token;
}

cl_event DataPipeline::subtract_mean(opencl::MemoryHandler *data,
                                     cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = data->size / sizeof(cl_float);
  std::cout << "Calcutating mean from " << len << " elements" << std::endl;
  u64 buf_sum = 0;
  // mean (NOTE:this is sync operation, cl_event actually does not matter)
  cl_event ev1 = sum(data, &buf_sum, ev_to_wait_for);
  float mean = ((float)buf_sum) / len;
  std::cout << "Mean: " << mean << std::endl;
  // subtract
  return subtract_from_all(data, mean, &ev1);
}

cl_event DataPipeline::sum(opencl::MemoryHandler *data, u64 *result,
                           cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = data->size / sizeof(cl_float);

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*_sum_kernel, global_work_size, local_work_size,
                            len, 1);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  *result = 0;
  if (!allocation_has_right_size(_tmp_64bit, sizeof(cl_ulong))) {
    _tmp_64bit = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  }
  _context->write_buffer(_tmp_64bit, (void *)result, true);  // zeroe

  // kernel args
  _sum_kernel->push_arg(data);
  _sum_kernel->push_arg(_tmp_64bit);
  _sum_kernel->push_arg(sizeof(cl_float) * local_work_size[0], nullptr);
  _sum_kernel->push_arg(sizeof(cl_uint), (void *)&len);

  // run
  cl_event finish_token = _sum_kernel->execute(1, global_work_size,
                                               local_work_size, ev_to_wait_for);

  // read (values may not be exactly the same since float->long data loss,
  // but should be close enough)
  _context->read_buffer(_tmp_64bit, 0, sizeof(cl_ulong), (void *)result, true,
                        &finish_token, 1);

  return finish_token;
}

cl_event DataPipeline::subtract_from_all(opencl::MemoryHandler *data, float val,
                                         cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = data->size / sizeof(cl_float);

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*_subtract_from_all_kernel, global_work_size,
                            local_work_size, len, 1);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  // kernel args
  _subtract_from_all_kernel->push_arg(data);
  _subtract_from_all_kernel->push_arg(sizeof(cl_float), (void *)&val);
  _subtract_from_all_kernel->push_arg(sizeof(cl_uint), (void *)&len);

  // run
  cl_event finish_token = _subtract_from_all_kernel->execute(
      1, global_work_size, local_work_size, ev_to_wait_for);

  return finish_token;
}

///
/// execute: cnn forward propagation
///

void DataPipeline::pre_execute_layer_validation(const LayerData &data,
                                                opencl::MemoryHandler *input,
                                                size_t input_w,
                                                size_t input_h) {
  LayerData::validate(data);

  size_t expected_input_size = data.input_size(input_w, input_h);
  if (expected_input_size > sizeof(cl_float) > input->size) {
    char buf[255];
    snprintf(buf, 255,
             "Declared input_w(%d)*input_h(%d)*n_prev_filter_cnt(%d)=%d "
             "is bigger then allocated gpu memory (%d elements).",
             input_w, input_h, data.n_prev_filter_cnt, expected_input_size,
             input->size);
    throw std::runtime_error(buf);
  }
}

cl_event DataPipeline::execute_layer(
    opencl::Kernel &kernel,                                               //
    const LayerData &data, cnn_sr::CnnLayerGpuAllocationPool &gpu_alloc,  //
    opencl::MemoryHandler *&gpu_buf_in, size_t input_w, size_t input_h,
    cl_event *ev_to_wait_for) {
  pre_execute_layer_validation(data, gpu_buf_in, input_w, input_h);

  size_t out_size[2];
  data.get_output_dimensions(out_size, input_w, input_h);
  size_t out_count = out_size[0] * out_size[1] * data.current_filter_count;
  std::cout << "out size: " << out_size[0] << "x" << out_size[1] << "x"
            << data.current_filter_count << "=" << out_count << std::endl;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size, input_w,
                            input_h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;

  // buffers: W, B, out_target
  size_t weights_alloc_size = sizeof(cl_float) * data.weight_size(),
         bias_alloc_size = sizeof(cl_float) * data.bias_size(),
         out_alloc_size = sizeof(cl_float) * out_count;

  if (!allocation_has_right_size(gpu_alloc.weights, weights_alloc_size)) {
    gpu_alloc.weights =
        _context->allocate(CL_MEM_READ_ONLY, weights_alloc_size);
    _context->write_buffer(gpu_alloc.weights, (void *)data.weights_ptr(), true);
  }
  if (!allocation_has_right_size(gpu_alloc.bias, bias_alloc_size)) {
    gpu_alloc.bias = _context->allocate(CL_MEM_READ_ONLY, bias_alloc_size);
    _context->write_buffer(gpu_alloc.bias, (void *)data.bias_ptr(), true);
  }
  if (!allocation_has_right_size(gpu_alloc.output, out_alloc_size)) {
    gpu_alloc.output = _context->allocate(CL_MEM_READ_WRITE, out_alloc_size);
  }
  _context->zeros_float(gpu_alloc.output, true);

  // args
  kernel.push_arg(gpu_buf_in);
  kernel.push_arg(gpu_alloc.output);
  kernel.push_arg(gpu_alloc.weights);
  kernel.push_arg(gpu_alloc.bias);
  kernel.push_arg(sizeof(cl_uint), (void *)&data.n_prev_filter_cnt);
  kernel.push_arg(sizeof(cl_uint), (void *)&data.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_h);

  // run
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  return kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

///
/// backpropagation
///

cl_event DataPipeline::mean_squared_error(
    opencl::MemoryHandler *gpu_buf_ground_truth,
    opencl::MemoryHandler *gpu_buf_algo_res,
    opencl::MemoryHandler *&gpu_buf_target,  //
    size_t ground_truth_w, size_t ground_truth_h, size_t total_padding,
    cl_event *ev_to_wait_for) {
  //
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t algo_w = ground_truth_w - total_padding,
         algo_h = ground_truth_h - total_padding, algo_size = algo_w * algo_h;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*_mse_kernel, global_work_size, local_work_size,
                            algo_w, algo_h);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  // check allocations
  /* clang-format off */
  if (!allocation_has_right_size(gpu_buf_ground_truth, sizeof(cl_float) * ground_truth_w * ground_truth_h)) {
    throw std::runtime_error(
        "Provided ground_truth_w, ground_truth_h dimensions did not match "
        "allocated gpu_buf_ground_truth buffer size");
  }
  if (!allocation_has_right_size(gpu_buf_algo_res, sizeof(cl_float) * algo_size)) {
    throw std::runtime_error( "Allocated gpu_buf_algo_res buffer size did not match calculated size");
  }
  if (!allocation_has_right_size(gpu_buf_target, sizeof(cl_float) * algo_size)) {
    gpu_buf_target = _context->allocate(CL_MEM_READ_WRITE, sizeof(cl_float) * algo_size);
  }
  /* clang-format on */
  _context->zeros_float(gpu_buf_target, true);

  // kernel args
  _mse_kernel->push_arg(gpu_buf_ground_truth);
  _mse_kernel->push_arg(gpu_buf_algo_res);
  _mse_kernel->push_arg(gpu_buf_target);
  _mse_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  _mse_kernel->push_arg(sizeof(cl_uint), (void *)&algo_w);
  _mse_kernel->push_arg(sizeof(cl_uint), (void *)&algo_size);

  // run
  return _mse_kernel->execute(1, global_work_size, local_work_size,
                              ev_to_wait_for);
}

cl_event DataPipeline::calculate_deltas(
    opencl::Kernel &kernel,       //
    const LayerData &prev_layer,  //
    const LayerData &curr_layer,  //
    CnnLayerGpuAllocationPool &prev_gpu_alloc,
    CnnLayerGpuAllocationPool &curr_gpu_alloc,  //
    size_t curr_layer_out_w,
    size_t curr_layer_out_h,  //
    cl_event *ev_to_wait_for) {
  //
  // @pre validation
  LayerData::validate(curr_layer);
  // TODO assert prev_layer.current_filter_count ==
  // current_layer.n_previous_filter_count

  size_t input_w = curr_layer_out_w + curr_layer.f_spatial_size - 1,
         input_h = curr_layer_out_h + curr_layer.f_spatial_size - 1;
  size_t out_count = curr_layer_out_w * curr_layer_out_h *
                     curr_layer.current_filter_count,
         in_count = input_w * input_h * curr_layer.n_prev_filter_cnt;
  // gpu memory alloc sizes
  size_t weights_alloc_size = sizeof(cl_float) * curr_layer.weight_size(),
         in_alloc_size = sizeof(cl_float) * in_count,
         out_alloc_size = sizeof(cl_float) * out_count;

  /* clang-format off */
  // gpu memory allocation, used buffers:
  //   curr_gpu_alloc.deltas
  //   curr_gpu_alloc.weights
  //   prev_layer.output <- this is input for current layer (used for activation_func_derivative)
  //   prev_layer.deltas <- as target
  if (!allocation_has_right_size(curr_gpu_alloc.deltas, out_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate deltas for previous layer, but deltas for current layer are not valid !");
  }
  if (!allocation_has_right_size(curr_gpu_alloc.weights, weights_alloc_size)) {
    curr_gpu_alloc.weights = _context->allocate(CL_MEM_READ_ONLY, weights_alloc_size);
    _context->write_buffer(curr_gpu_alloc.weights, (void *)curr_layer.weights_ptr(), true);
  }
  if (!allocation_has_right_size(prev_gpu_alloc.output, in_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate deltas for previous layer, but there are no previous layer output values."
        "They are normally allocated during forward step.");
  }
  if (!allocation_has_right_size(prev_gpu_alloc.deltas, in_alloc_size)) {
    prev_gpu_alloc.deltas = _context->allocate(CL_MEM_READ_WRITE, in_alloc_size);
  }
  _context->zeros_float(prev_gpu_alloc.deltas, true);
  /* clang-format on */

  // args
  kernel.push_arg(curr_gpu_alloc.deltas);
  kernel.push_arg(prev_gpu_alloc.output);
  kernel.push_arg(prev_gpu_alloc.deltas);
  kernel.push_arg(curr_gpu_alloc.weights);
  kernel.push_arg(sizeof(cl_uint), (void *)&prev_layer.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&curr_layer.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&curr_layer.current_filter_count);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_h);

  _context->block();  // TODO remove

  // run
  size_t global_work_size[2], local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size,  //
                            input_w, input_h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  return kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

void DataPipeline::backpropagate(opencl::Kernel &kernel,  //
                                 LayerData &layer_data,   //
                                 opencl::MemoryHandler *layer_input,
                                 CnnLayerGpuAllocationPool &gpu_alloc,
                                 size_t layer_out_w, size_t layer_out_h,
                                 cl_event *ev_to_wait_for) {
  LayerData::validate(layer_data);

  size_t input_w = layer_out_w + layer_data.f_spatial_size - 1,
         input_h = layer_out_h + layer_data.f_spatial_size - 1;
  size_t out_count =
             layer_out_w * layer_out_h * layer_data.current_filter_count,
         in_count = input_w * input_h * layer_data.n_prev_filter_cnt;

  // due too local scrath buffer there is a lot of calculations to be done
  size_t global_work_size[2], local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size,  //
                            layer_out_w, layer_out_h);
  size_t local_size = local_work_size[0] * local_work_size[1],
         groups_count = (global_work_size[0] / local_work_size[0]) *
                        (global_work_size[1] / local_work_size[1]);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;
  std::cout << "Total " << groups_count << " groups, " << local_size
            << " work items each" << std::endl;

  // gpu memory alloc sizes
  /* clang-format off */
  size_t per_filter_size = layer_data.f_spatial_size * layer_data.f_spatial_size * layer_data.n_prev_filter_cnt,
         grad_w_elems = groups_count * layer_data.current_filter_count * per_filter_size,
         grad_b_elems = groups_count * layer_data.current_filter_count,
         in_alloc_size = sizeof(cl_float) * in_count,
         out_alloc_size = sizeof(cl_float) * out_count,
         grad_w_size = sizeof(cl_float) * grad_w_elems,
         grad_b_size = sizeof(cl_float) * grad_b_elems,
         scratch_w_size = sizeof(cl_float) * local_size * per_filter_size,
         scratch_b_size = sizeof(cl_float) * local_size * layer_data.current_filter_count;
  /* clang-format on */

  /* clang-format off */
  if (!allocation_has_right_size(gpu_alloc.deltas, out_alloc_size)) {
    throw std::runtime_error("Tried to calculate gradients, but deltas for current layer are not valid");
  }
  if (!allocation_has_right_size(layer_input, in_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate gradients, but there are no previous layer output values."
        "They are normally allocated during forward step.");
  }
  /* clang-format on */
  if (!allocation_has_right_size(gpu_alloc.grad_w, grad_w_size)) {
    gpu_alloc.grad_w = _context->allocate(CL_MEM_READ_WRITE, grad_w_size);
  }
  if (!allocation_has_right_size(gpu_alloc.grad_b, grad_b_size)) {
    gpu_alloc.grad_b = _context->allocate(CL_MEM_READ_WRITE, grad_b_size);
  }
  // this is actualy not needed, but if I'm mistaken it will panic
  _context->zeros_float(gpu_alloc.grad_w, true);
  _context->zeros_float(gpu_alloc.grad_b, true);

  // args
  kernel.push_arg(gpu_alloc.deltas);
  kernel.push_arg(layer_input);
  kernel.push_arg(gpu_alloc.grad_w);
  kernel.push_arg(gpu_alloc.grad_b);
  kernel.push_arg(scratch_w_size, nullptr);  // scratch buffer
  kernel.push_arg(scratch_b_size, nullptr);  // scratch buffer
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_data.n_prev_filter_cnt);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_data.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_out_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_out_h);

  _context->block();  // TODO remove

  // run
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  auto kernel_done_event =
      kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                     events_to_wait_for_count);

  std::cout << "Scheduled gpu backpropagation kernel, blocking" << std::endl;
  _context->block();
  std::cout << "kernel executed, doing cpu backpropagation step" << std::endl;

  std::vector<float> subresults_w(grad_w_elems);
  std::vector<float> subresults_b(grad_b_elems);
  _context->read_buffer(gpu_alloc.grad_w, (void *)&subresults_w[0], true,
                        &kernel_done_event, 1);
  _context->read_buffer(gpu_alloc.grad_b, (void *)&subresults_b[0], true,
                        &kernel_done_event, 1);
  // clear gradient results
  size_t bs = layer_data.bias_size();
  size_t ws = layer_data.weight_size();
  for (size_t j = 0; j < ws; j++) {
    layer_data.grad_weights[j] = 0;
  }
  for (size_t j = 0; j < bs; j++) {
    layer_data.grad_bias[j] = 0;
  }
  // summation
  for (size_t i = 0; i < groups_count; i++) {
    for (size_t j = 0; j < ws; j++) {
      // std::cout << "w " << subresults_w[i * ws + j] << std::endl;
      layer_data.grad_weights[j] += subresults_w[i * ws + j];
    }
    for (size_t j = 0; j < bs; j++) {
      // std::cout << "b " << subresults_b[i * bs + j] << std::endl;
      layer_data.grad_bias[j] += subresults_b[i * bs + j];
    }
  }
}

// end: namespace cnn_sr
}
