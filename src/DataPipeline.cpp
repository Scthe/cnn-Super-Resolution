#include "DataPipeline.hpp"

#include <stdexcept>  // std::runtime_error
#include <cstdio>     // snprintf

#include "LayerData.hpp"
#include "opencl/Context.hpp"
#include "opencl/UtilsOpenCL.hpp"

const bool print_work_dimensions = false;

std::string kernel_folder = "src/kernel/";

const char *const luma_kernel_file = "extract_luma.cl";
const char *const swap_luma_kernel_file = "swap_luma.cl";
const char *const squared_error_kernel_file = "squared_error.cl";
const char *const sum_kernel_file = "sum.cl";
// forward:
const char *const layer_kernel_file = "layer_uber_kernel.cl";
// backpropagation:
const char *const deltas_kernel_file = "layer_deltas.cl";
const char *const last_layer_delta_kernel_file = "last_layer_delta.cl";
const char *const backpropagate_kernel_file = "backpropagate.cl";
const char *const subtract_from_all_kernel_file = "subtract_from_all.cl";
const char *const update_parameters_kernel_file = "update_parameters.cl";

using namespace cnn_sr;

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
    : _context(context), _initialized(false) {}

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

#define ALLOCATION_HAS_RIGHT_SIZE(ALLOC_VAR, ALLOC_SIZE)              \
  (this->allocation_has_right_size__(ALLOC_VAR, ALLOC_SIZE, __LINE__, \
                                     STRINGIFY(ALLOC_VAR)))

bool DataPipeline::allocation_has_right_size__(opencl::MemoryHandle alloc,
                                               size_t size, size_t line,
                                               const char *variable_name) {
  if (alloc == gpu_nullptr) {
    // std::cout << variable_name << " is NULL" << std::endl;
    return false;
  }
  auto raw_mem = _context->raw_memory(alloc);
  if (raw_mem->size >= size) return true;

  std::cout << "Was forced to realocate gpu buffer. This is not optimal and "
               "may be a bug. In many cases DataPipeline is able to allocate "
               "buffer of right size, so You only need to explictly set "
               "MemoryHandle to gpu_nullptr. "  //
               "Expected: " << size << ", got: " << raw_mem->size
            << ". Code line: " << line << ", variable: '" << variable_name
            << "'" << std::endl;
  throw std::runtime_error(
      "Was forced to realocate gpu buffer due too difference in sizes.");
  return false;
}

size_t DataPipeline::element_count(opencl::MemoryHandle alloc, size_t el_size) {
  auto raw_mem = _context->raw_memory(alloc);
  return raw_mem->size / el_size;
}

opencl::Context *DataPipeline::context() { return _context; }

///
/// misc
///
void DataPipeline::print_buffer(opencl::MemoryHandle mh, const char *const name,
                                size_t lines) {
  auto raw = _context->raw_memory(mh);
  size_t len = raw->size / sizeof(cl_float);
  // std::cout << "len:" << len << std::endl;

  // read
  std::vector<float> data(len);
  _context->block();
  _context->read_buffer(mh, &data[0], true);

  // print
  std::cout << name << ": [" << std::endl;
  cnn_sr::utils::dump_vector(std::cout, data, "", len / lines, true);
  std::cout << "]" << std::endl
            << std::endl
            << std::endl;
}

///
/// Kernel loading
///

void DataPipeline::load_kernels(int load_flags) {
  bool load_luma = (load_flags & DataPipeline::LOAD_KERNEL_LUMA) != 0,
       load_back = (load_flags & DataPipeline::LOAD_KERNEL_BACKPROPAGATE) != 0,
       load_misc = (load_flags & DataPipeline::LOAD_KERNEL_MISC) != 0;

/* clang-format off */
#define ck(file, args, name) _context->create_kernel((kernel_folder + file).c_str(), args, name)

  if (load_luma) {
    auto norm_arg = "-D NORMALIZE";
    if (!_luma_kernel_norm)
      _luma_kernel_norm = ck(luma_kernel_file,      norm_arg, "extract_luma");
    if (!_luma_kernel_raw)
      _luma_kernel_raw =  ck(luma_kernel_file,      nullptr,  "extract_luma");
    if (!_swap_luma_kernel)
      _swap_luma_kernel = ck(swap_luma_kernel_file, nullptr,  "swap_luma");
  }

  if (load_misc) {
    if (!_squared_error_kernel)
      _squared_error_kernel       = ck(squared_error_kernel_file, nullptr, "squared_err");
    if (!_sum_kernel) _sum_kernel = ck(sum_kernel_file,           nullptr, "sum");
    if (!_sum_squared_kernel)
      _sum_squared_kernel         = ck(sum_kernel_file, "-D SUM_SQUARED", "sum");
    if (!_subtract_from_all_kernel)
      _subtract_from_all_kernel   = ck(subtract_from_all_kernel_file, nullptr, "sub_from_all");
  }

  if (load_back) {
    if (!_last_layer_delta_kernel)
      _last_layer_delta_kernel  = ck(last_layer_delta_kernel_file, nullptr, "last_layer_delta");
    if (!_update_parameters_kernel)
      _update_parameters_kernel = ck(update_parameters_kernel_file, nullptr, "update_params");
    if (!_backpropagate_kernel)
      _backpropagate_kernel     = ck(backpropagate_kernel_file,     nullptr, "backpropagate");
    /* clang-format on */
  }
#undef ck
}

opencl::Kernel *DataPipeline::create_layer_kernel(const LayerData &d,
                                                  bool skip_relu) {
  char buf[255];
  std::string defs =
      "-D CURRENT_FILTER_COUNT=%d -D PREVIOUS_FILTER_COUNT=%d -D "
      "F_SPATIAL_SIZE=%d";
  if (skip_relu) defs += " -D SKIP_RELU";

  snprintf(buf, 255, defs.c_str(), d.current_filter_count, d.n_prev_filter_cnt,
           d.f_spatial_size);
  return _context->create_kernel((kernel_folder + layer_kernel_file).c_str(),
                                 buf, "forward");
}

opencl::Kernel *DataPipeline::create_deltas_kernel(const LayerData &d) {
  char buf[255];
  snprintf(buf, 255, "-D CURRENT_FILTER_COUNT=%d", d.current_filter_count);
  return _context->create_kernel((kernel_folder + deltas_kernel_file).c_str(),
                                 buf, "deltas");
}

///
/// execute: misc
///

cl_event DataPipeline::extract_luma(opencl::utils::ImageData &img_data,
                                    opencl::MemoryHandle &gpu_buf_raw_img,
                                    opencl::MemoryHandle &gpu_buf_luma,
                                    bool normalize, cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_LUMA);

  size_t out_pixel_count = img_data.w * img_data.h /* sizeof(cl_char)*/;
  auto kernel = normalize ? _luma_kernel_norm : _luma_kernel_raw;

  // memory allocation
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_raw_img, out_pixel_count)) {
    gpu_buf_raw_img = _context->create_image(
        CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  }
  _context->write_image(gpu_buf_raw_img, img_data, true);
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_luma, out_pixel_count)) {
    gpu_buf_luma = _context->allocate(CL_MEM_READ_WRITE,
                                      sizeof(cl_float) * out_pixel_count);
  }

  // kernel args
  kernel->push_arg(gpu_buf_raw_img);
  kernel->push_arg(gpu_buf_luma);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.h);

  // Launch kernel
  size_t global_work_size[2], local_work_size[2],
      work_dims[2] = {(size_t)img_data.w, (size_t)img_data.h};
  opencl::utils::work_sizes(*kernel, 2, global_work_size, local_work_size,
                            work_dims, print_work_dimensions);
  auto finish_token = kernel->execute(2, global_work_size, local_work_size,
                                      ev_to_wait_for, ev_to_wait_for ? 1 : 0);
  return finish_token;
}

cl_event DataPipeline::swap_luma(opencl::utils::ImageData &img_data,
                                 opencl::MemoryHandle &gpu_buf_org_img,
                                 opencl::MemoryHandle gpu_buf_new_luma,
                                 opencl::MemoryHandle &target,
                                 size_t new_luma_w, size_t new_luma_h,
                                 cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_LUMA);

  size_t img_size = img_data.w * img_data.h /* sizeof(cl_char)*/,
         img_size_3ch = img_size * 3,
         new_luma_size = new_luma_w * new_luma_h * sizeof(cl_float);

  // memory allocation
  // (writing image may be redundant, but let's ignore this)
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_new_luma, new_luma_size)) {
    throw std::runtime_error("Invalid size of new luma buffer");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(target, img_size_3ch)) {
    target = _context->allocate(CL_MEM_READ_WRITE, img_size_3ch);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_org_img, img_size * 4)) {
    gpu_buf_org_img = _context->create_image(
        CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  }
  _context->write_image(gpu_buf_org_img, img_data, true);

  // kernel args
  _swap_luma_kernel->push_arg(gpu_buf_org_img);
  _swap_luma_kernel->push_arg(gpu_buf_new_luma);
  _swap_luma_kernel->push_arg(target);
  _swap_luma_kernel->push_arg(sizeof(cl_uint), (void *)&img_data.w);
  _swap_luma_kernel->push_arg(sizeof(cl_uint), (void *)&img_data.h);
  _swap_luma_kernel->push_arg(sizeof(cl_uint), (void *)&new_luma_w);
  _swap_luma_kernel->push_arg(sizeof(cl_uint), (void *)&new_luma_h);

  // Launch kernel
  size_t global_work_size[2], local_work_size[2],
      work_dims[2] = {(size_t)img_data.w, (size_t)img_data.h};
  opencl::utils::work_sizes(*_swap_luma_kernel, 2, global_work_size,
                            local_work_size, work_dims, print_work_dimensions);
  auto finish_token =
      _swap_luma_kernel->execute(2, global_work_size, local_work_size,
                                 ev_to_wait_for, ev_to_wait_for ? 1 : 0);
  return finish_token;
}

cl_event DataPipeline::subtract_mean(opencl::MemoryHandle data, float *mean,
                                     cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = element_count(data, sizeof(cl_float));

  // std::cout << "Calcutating mean from " << len << " elements" << std::endl;
  auto buf_sum = sum(data, ev_to_wait_for);
  float mean_v = ((float)buf_sum) / len;
  if (mean) *mean = mean_v;
  // std::cout << "Mean: " << mean << std::endl;
  // subtract
  return subtract_from_all(data, mean_v);
}

float DataPipeline::sum(opencl::MemoryHandle data, bool squared,
                        cl_event *ev_to_wait_for) {
  if (cnn_sr::warn_about_blocking_operation)
    std::cout << "BLOCK: sum" << std::endl;
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = element_count(data, sizeof(cl_float));
  auto *kernel = squared ? _sum_squared_kernel : _sum_kernel;

  float result = 0;
  if (!ALLOCATION_HAS_RIGHT_SIZE(_tmp_gpu_float, sizeof(cl_float))) {
    _tmp_gpu_float = _context->allocate(CL_MEM_READ_WRITE, sizeof(cl_float));
  }
  _context->write_buffer(_tmp_gpu_float, (void *)&result, true);  // zeroe

  size_t global_work_size, local_work_size;
  opencl::utils::work_sizes(*kernel, 1, &global_work_size, &local_work_size,
                            &len, print_work_dimensions);
  // kernel args
  kernel->push_arg(data);
  kernel->push_arg(_tmp_gpu_float);
  kernel->push_arg(sizeof(cl_float) * local_work_size, nullptr);
  kernel->push_arg(sizeof(cl_uint), (void *)&len);

  // run
  cl_event finish_token =
      kernel->execute(1, &global_work_size, &local_work_size, ev_to_wait_for);

  // read and return result
  _context->read_buffer(_tmp_gpu_float, (void *)&result, true, &finish_token,
                        1);
  return result;
}

cl_event DataPipeline::subtract_from_all(opencl::MemoryHandle data, float val,
                                         cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t len = element_count(data, sizeof(cl_float));

  // kernel args
  _subtract_from_all_kernel->push_arg(data);
  _subtract_from_all_kernel->push_arg(sizeof(cl_float), (void *)&val);
  _subtract_from_all_kernel->push_arg(sizeof(cl_uint), (void *)&len);

  // run
  size_t global_work_size, local_work_size;
  opencl::utils::work_sizes(*_subtract_from_all_kernel, 1, &global_work_size,
                            &local_work_size, &len, print_work_dimensions);
  cl_event finish_token = _subtract_from_all_kernel->execute(
      1, &global_work_size, &local_work_size, ev_to_wait_for);

  return finish_token;
}

///
/// execute: cnn forward propagation
///

void DataPipeline::pre_execute_layer_validation(const LayerData &data,
                                                opencl::MemoryHandle input,
                                                size_t input_w,
                                                size_t input_h) {
  LayerData::validate(data);

  size_t expected_input_size = data.input_size(input_w, input_h),
         cnt = element_count(input, sizeof(cl_float));
  if (expected_input_size > sizeof(cl_float) * cnt) {
    char buf[255];
    snprintf(buf, 255,
             "Declared input_w(%d)*input_h(%d)*n_prev_filter_cnt(%d)=%d "
             "is bigger then allocated gpu memory (%d elements).",
             input_w, input_h, data.n_prev_filter_cnt, expected_input_size,
             cnt);
    throw std::runtime_error(buf);
  }
}

cl_event DataPipeline::execute_layer(opencl::Kernel &kernel,
                                     const LayerData &data,
                                     LayerAllocationPool &gpu_alloc,  //
                                     opencl::MemoryHandle &gpu_buf_in,
                                     size_t input_w, size_t input_h,
                                     opencl::MemoryHandle &gpu_buf_out,
                                     cl_event *ev_to_wait_for) {
  pre_execute_layer_validation(data, gpu_buf_in, input_w, input_h);

  size_t out_size[2];
  data.get_output_dimensions(out_size, input_w, input_h);
  size_t out_count = out_size[0] * out_size[1] * data.current_filter_count;
  // std::cout << "out size: " << out_size[0] << "x" << out_size[1] << "x"
  // << data.current_filter_count << "=" << out_count << std::endl;

  // buffers: W, B, out_target
  size_t weights_alloc_size = sizeof(cl_float) * data.weight_size(),
         bias_alloc_size = sizeof(cl_float) * data.bias_size(),
         out_alloc_size = sizeof(cl_float) * out_count;

  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.weights, weights_alloc_size)) {
    gpu_alloc.weights =
        _context->allocate(CL_MEM_READ_WRITE, weights_alloc_size);
    _context->write_buffer(gpu_alloc.weights, (void *)data.weights_ptr(), true);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.bias, bias_alloc_size)) {
    gpu_alloc.bias = _context->allocate(CL_MEM_READ_WRITE, bias_alloc_size);
    _context->write_buffer(gpu_alloc.bias, (void *)data.bias_ptr(), true);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_out, out_alloc_size)) {
    gpu_buf_out = _context->allocate(CL_MEM_READ_WRITE, out_alloc_size);
  }

  // args
  kernel.push_arg(gpu_buf_in);
  kernel.push_arg(gpu_buf_out);
  kernel.push_arg(gpu_alloc.weights);
  kernel.push_arg(gpu_alloc.bias);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&input_h);

  // run
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  size_t global_work_size[2], local_work_size[2],
      work_dims[2] = {input_w, input_h};  // TODO output_w,output_h ?
  opencl::utils::work_sizes(kernel, 2, global_work_size, local_work_size,
                            work_dims, print_work_dimensions);
  return kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

///
/// backpropagation
///

cl_event DataPipeline::squared_error(opencl::MemoryHandle gpu_buf_ground_truth,
                                     size_t ground_truth_w,  //
                                     size_t ground_truth_h,  //
                                     size_t sample_id,
                                     opencl::MemoryHandle gpu_buf_algo_res,
                                     opencl::MemoryHandle tmp_buffer,
                                     float &target, size_t total_padding,
                                     cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t algo_w = ground_truth_w - total_padding,
         algo_h = ground_truth_h - total_padding,  //
      algo_size = algo_w * algo_h;

  // check allocations
  /* clang-format off */
  /*
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_ground_truth, sizeof(cl_float) * ground_truth_w * ground_truth_h)) {
    throw std::runtime_error(
        "Provided ground_truth_w, ground_truth_h dimensions did not match "
        "allocated gpu_buf_ground_truth buffer size");
  }*/
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_algo_res, sizeof(cl_float) * algo_size)) {
    throw std::runtime_error( "Allocated gpu_buf_algo_res buffer size did not match calculated size");
  }
  /* clang-format on */
  float zero = 0.0f;
  auto event_count = ev_to_wait_for == nullptr ? 0 : 1;
  auto ev_write = _context->write_buffer(tmp_buffer, (void *)&zero, false,
                                         ev_to_wait_for, event_count);
  ev_to_wait_for = &ev_write;

  size_t global_work_size[2], local_work_size[2],
      work_dims[2] = {algo_w, algo_h};
  opencl::utils::work_sizes(*_squared_error_kernel, 2, global_work_size,
                            local_work_size, work_dims, print_work_dimensions);

  // kernel args
  size_t local_mem_size = local_work_size[0] * local_work_size[1];
  _squared_error_kernel->push_arg(gpu_buf_ground_truth);
  _squared_error_kernel->push_arg(gpu_buf_algo_res);
  _squared_error_kernel->push_arg(tmp_buffer);
  _squared_error_kernel->push_arg(sizeof(cl_float) * local_mem_size,
                                  nullptr);  // scratch
  _squared_error_kernel->push_arg(sizeof(cl_uint), (void *)&sample_id);
  _squared_error_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  _squared_error_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_h);
  _squared_error_kernel->push_arg(sizeof(cl_uint), (void *)&algo_w);
  _squared_error_kernel->push_arg(sizeof(cl_uint), (void *)&algo_h);

  // run
  cl_event finish_token = _squared_error_kernel->execute(
      2, global_work_size, local_work_size, ev_to_wait_for);

  return _context->read_buffer(tmp_buffer, (void *)&target, false,
                               &finish_token, 1);
}

cl_event DataPipeline::last_layer_delta(
    opencl::MemoryHandle gpu_buf_ground_truth,  //
    size_t ground_truth_w, size_t ground_truth_h, size_t sample_id,
    opencl::MemoryHandle gpu_buf_algo_res,
    opencl::MemoryHandle &gpu_buf_target,  //
    size_t total_padding, cl_event *ev_to_wait_for) {
  //
  check_initialized(DataPipeline::LOAD_KERNEL_BACKPROPAGATE);
  size_t algo_w = ground_truth_w - total_padding,
         algo_h = ground_truth_h - total_padding,  //
      algo_size = algo_w * algo_h;

  // check allocations
  /* clang-format off */
  /*
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_ground_truth, sizeof(cl_float) * ground_truth_w * ground_truth_h)) {
    throw std::runtime_error(
        "Provided ground_truth_w, ground_truth_h dimensions did not match "
        "allocated gpu_buf_ground_truth buffer size");
  }*/
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_algo_res, sizeof(cl_float) * algo_size)) {
    throw std::runtime_error( "Allocated gpu_buf_algo_res buffer size did not match calculated size");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_buf_target, sizeof(cl_float) * algo_size)) {
    gpu_buf_target = _context->allocate(CL_MEM_READ_WRITE, sizeof(cl_float) * algo_size);
  }
  /* clang-format on */

  // kernel args
  _last_layer_delta_kernel->push_arg(gpu_buf_ground_truth);
  _last_layer_delta_kernel->push_arg(gpu_buf_algo_res);
  _last_layer_delta_kernel->push_arg(gpu_buf_target);
  _last_layer_delta_kernel->push_arg(sizeof(cl_uint), (void *)&sample_id);
  _last_layer_delta_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  _last_layer_delta_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_h);
  _last_layer_delta_kernel->push_arg(sizeof(cl_uint), (void *)&algo_w);
  _last_layer_delta_kernel->push_arg(sizeof(cl_uint), (void *)&algo_h);

  // run
  size_t global_work_size[2], local_work_size[2],
      work_dims[2] = {algo_w, algo_h};
  opencl::utils::work_sizes(*_last_layer_delta_kernel, 2, global_work_size,
                            local_work_size, work_dims, print_work_dimensions);
  return _last_layer_delta_kernel->execute(2, global_work_size, local_work_size,
                                           ev_to_wait_for);
}

cl_event DataPipeline::calculate_deltas(
    opencl::Kernel &kernel,  //
    const LayerData &curr_layer, const LayerData &next_layer,
    LayerAllocationPool &next_gpu_alloc,  //
    opencl::MemoryHandle curr_deltas, opencl::MemoryHandle next_deltas,
    size_t next_layer_out_w, size_t next_layer_out_h, size_t sample_id,
    opencl::MemoryHandle curr_output, cl_event *ev_to_wait_for) {
  //
  // @pre validation
  LayerData::validate(next_layer);
  if (curr_layer.current_filter_count != next_layer.n_prev_filter_cnt) {
    throw std::runtime_error(
        "When calculating deltas for layer it's filter count should be equal "
        "to next layer's previous filter count");
  }

  size_t out_w = next_layer_out_w + next_layer.f_spatial_size - 1,
         out_h = next_layer_out_h + next_layer.f_spatial_size - 1;
  size_t next_out_size = next_layer_out_w * next_layer_out_h *
                         next_layer.current_filter_count,
         out_size = out_w * out_h * next_layer.n_prev_filter_cnt;
  // gpu memory alloc sizes
  size_t weights_alloc_size = sizeof(cl_float) * next_layer.weight_size(),
         out_alloc_size = sizeof(cl_float) * out_size,
         next_out_alloc_size = sizeof(cl_float) * next_out_size;

  /* clang-format off */
  // gpu memory allocation, used buffers:
  //   next_gpu_alloc.deltas
  //   next_gpu_alloc.weights
  //   curr_layer.output <- this is input for current layer (used for activation_func_derivative)
  //   curr_layer.deltas <- as target
  /*
  if (!ALLOCATION_HAS_RIGHT_SIZE(next_deltas, next_out_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate deltas for previous layer, but deltas for current layer are not valid !");
  }*/
  if (!ALLOCATION_HAS_RIGHT_SIZE(next_gpu_alloc.weights, weights_alloc_size)) {
    next_gpu_alloc.weights = _context->allocate(CL_MEM_READ_WRITE, weights_alloc_size);
    _context->write_buffer(next_gpu_alloc.weights, (void *)next_layer.weights_ptr(), true);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(curr_output, out_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate deltas for previous layer, but there are no previous layer output values."
        "They are normally allocated during forward step.");
  }
  /*
  if (!ALLOCATION_HAS_RIGHT_SIZE(curr_deltas, out_alloc_size)) {
    curr_deltas = _context->allocate(CL_MEM_READ_WRITE, out_alloc_size);
  }*/
  /* clang-format on */

  // args
  kernel.push_arg(next_deltas);
  kernel.push_arg(curr_output);
  kernel.push_arg(curr_deltas);  // target
  kernel.push_arg(next_gpu_alloc.weights);
  kernel.push_arg(sizeof(cl_uint), (void *)&sample_id);
  kernel.push_arg(sizeof(cl_uint), (void *)&curr_layer.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&next_layer.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&next_layer.current_filter_count);
  kernel.push_arg(sizeof(cl_uint), (void *)&out_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&out_h);

  // run
  size_t global_work_size[2], local_work_size[2], work_dims[2] = {out_w, out_h};
  opencl::utils::work_sizes(kernel, 2, global_work_size, local_work_size,
                            work_dims, print_work_dimensions);

  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  return kernel.execute(2, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

cl_event DataPipeline::backpropagate(LayerData &layer_data,  //
                                     opencl::MemoryHandle layer_input,
                                     opencl::MemoryHandle layer_deltas,
                                     LayerAllocationPool &gpu_alloc,
                                     size_t layer_out_w, size_t layer_out_h,
                                     size_t sample_id,  //
                                     cl_event *ev_to_wait_for, size_t ev_cnt) {
  LayerData::validate(layer_data);
  check_initialized(DataPipeline::LOAD_KERNEL_BACKPROPAGATE);

  size_t input_w = layer_out_w + layer_data.f_spatial_size - 1,
         input_h = layer_out_h + layer_data.f_spatial_size - 1;
  size_t out_count =
             layer_out_w * layer_out_h * layer_data.current_filter_count,
         in_count = input_w * input_h * layer_data.n_prev_filter_cnt;

  size_t weights_size = layer_data.f_spatial_size * layer_data.f_spatial_size *
                        layer_data.n_prev_filter_cnt *
                        layer_data.current_filter_count;
  // std::cout << "weights_size: " << weights_size << std::endl;

  // allocations
  size_t in_alloc_size = sizeof(cl_float) * in_count,
         out_alloc_size = sizeof(cl_float) * out_count,
         grad_w_size = sizeof(cl_float) * layer_data.weight_size(),
         grad_b_size = sizeof(cl_float) * layer_data.bias_size();
  /* clang-format off */
  /*
  if (!ALLOCATION_HAS_RIGHT_SIZE(layer_deltas, out_alloc_size)) {
    throw std::runtime_error("Tried to calculate gradients, but deltas for current layer are not valid");
  }*/
  if (!ALLOCATION_HAS_RIGHT_SIZE(layer_input, in_alloc_size)) {
    throw std::runtime_error(
        "Tried to calculate gradients, but there are no previous layer output values."
        "They are normally allocated during forward step.");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.accumulating_grad_w, grad_w_size)) {
    gpu_alloc.accumulating_grad_w = _context->allocate(CL_MEM_READ_WRITE, grad_w_size);
    _context->zeros_float(gpu_alloc.accumulating_grad_w, true);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.accumulating_grad_b, grad_b_size)) {
    gpu_alloc.accumulating_grad_b = _context->allocate(CL_MEM_READ_WRITE, grad_b_size);
    _context->zeros_float(gpu_alloc.accumulating_grad_b, true);
  }
  /* clang-format on */

  // args
  opencl::Kernel &kernel = *_backpropagate_kernel;
  kernel.push_arg(layer_deltas);
  kernel.push_arg(layer_input);
  kernel.push_arg(gpu_alloc.accumulating_grad_w);
  kernel.push_arg(gpu_alloc.accumulating_grad_b);
  kernel.push_arg(sizeof(cl_uint), (void *)&sample_id);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_data.current_filter_count);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_data.n_prev_filter_cnt);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_data.f_spatial_size);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_out_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&layer_out_h);

  // run
  size_t global_work_size[3], local_work_size[3];
  opencl::utils::work_sizes(kernel, 1, global_work_size, local_work_size,
                            &weights_size, print_work_dimensions);
  int events_to_wait_for_count = !ev_to_wait_for ? 0 : ev_cnt == 0 ? 1 : ev_cnt;
  return kernel.execute(1, global_work_size, local_work_size, ev_to_wait_for,
                        events_to_wait_for_count);
}

cl_event DataPipeline::update_parameters(LayerData &layer_data,  //
                                         LayerAllocationPool &gpu_alloc,
                                         size_t batch_size, float momentum,
                                         float w_decay, float learning_rate,
                                         cl_event *ev_to_wait_for) {
  LayerData::validate(layer_data);
  check_initialized(DataPipeline::LOAD_KERNEL_BACKPROPAGATE);

  size_t weights_size = layer_data.weight_size(),
         bias_size = layer_data.bias_size(),
         weights_alloc_size = sizeof(cl_float) * weights_size,
         bias_alloc_size = sizeof(cl_float) * bias_size;

  // allocations
  /* clang-format off */
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.weights, weights_alloc_size)) {
    throw std::runtime_error("Tried to update weights, but old values are not valid. "
                             "Impossible if forward pass was completed");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.bias, bias_alloc_size)) {
    throw std::runtime_error("Tried to update bias, but old values are not valid. "
                             "Impossible if forward pass was completed");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.accumulating_grad_w, weights_alloc_size)) {
    throw std::runtime_error("Tried to update weights, but gradient values are not valid. "
                             "Impossible if backpropagation was completed");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.accumulating_grad_b, bias_alloc_size)) {
    throw std::runtime_error("Tried to update bias, but gradient values are not valid. "
                             "Impossible if backpropagation was completed");
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.previous_batch_delta_w, weights_alloc_size)) {
    gpu_alloc.previous_batch_delta_w = _context->allocate(CL_MEM_READ_WRITE, weights_alloc_size);
    _context->zeros_float(gpu_alloc.previous_batch_delta_w, true);
  }
  if (!ALLOCATION_HAS_RIGHT_SIZE(gpu_alloc.previous_batch_delta_b, bias_alloc_size)) {
    gpu_alloc.previous_batch_delta_b = _context->allocate(CL_MEM_READ_WRITE, bias_alloc_size);
    _context->zeros_float(gpu_alloc.previous_batch_delta_b, true);
  }
  /* clang-format on */

  // args
  _update_parameters_kernel->push_arg(gpu_alloc.weights);
  _update_parameters_kernel->push_arg(gpu_alloc.bias);
  _update_parameters_kernel->push_arg(gpu_alloc.accumulating_grad_w);
  _update_parameters_kernel->push_arg(gpu_alloc.accumulating_grad_b);
  _update_parameters_kernel->push_arg(gpu_alloc.previous_batch_delta_w);
  _update_parameters_kernel->push_arg(gpu_alloc.previous_batch_delta_b);
  _update_parameters_kernel->push_arg(sizeof(cl_float), (void *)&momentum);
  _update_parameters_kernel->push_arg(sizeof(cl_float), (void *)&w_decay);
  _update_parameters_kernel->push_arg(sizeof(cl_float), (void *)&learning_rate);
  _update_parameters_kernel->push_arg(sizeof(cl_uint), (void *)&batch_size);
  _update_parameters_kernel->push_arg(sizeof(cl_uint), (void *)&weights_size);
  _update_parameters_kernel->push_arg(sizeof(cl_uint), (void *)&bias_size);

  // run
  int events_to_wait_for_count = ev_to_wait_for ? 1 : 0;
  size_t global_work_size, local_work_size;
  opencl::utils::work_sizes(*_update_parameters_kernel, 1, &global_work_size,
                            &local_work_size, &weights_size,
                            print_work_dimensions);
  return _update_parameters_kernel->execute(1, &global_work_size,
                                            &local_work_size, ev_to_wait_for,
                                            events_to_wait_for_count);
}

// end: namespace cnn_sr
}
