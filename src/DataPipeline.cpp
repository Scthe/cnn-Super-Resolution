#include "DataPipeline.hpp"

#include <stdexcept>  // std::runtime_error
#include <cstdio>     // snprintf
#include <memory>     // std::unique_ptr

#include "LayerData.hpp"
#include "opencl/Context.hpp"
#include "opencl/UtilsOpenCL.hpp"
#include "Config.hpp"

const char *const luma_kernel_file = "src/kernel/extract_luma.cl";
const char *const layer_kernel_file = "src/kernel/layer_uber_kernel.cl";
const char *const sum_sq_kernel_file = "src/kernel/sum_squared.cl";
const char *const sum_kernel_file = "src/kernel/sum.cl";
const char *const subtract_from_all_kernel_file =
    "src/kernel/subtract_from_all.cl";

namespace cnn_sr {

int DataPipeline::LOAD_KERNEL_LUMA = 1;
int DataPipeline::LOAD_KERNEL_LAYERS = 2;
int DataPipeline::LOAD_KERNEL_MISC = 4;
int DataPipeline::LOAD_KERNEL_NONE = 0;
int DataPipeline::LOAD_KERNEL_ALL = DataPipeline::LOAD_KERNEL_LUMA |
                                    DataPipeline::LOAD_KERNEL_LAYERS |
                                    DataPipeline::LOAD_KERNEL_MISC;

DataPipeline::DataPipeline(Config *cfg, opencl::Context *context)
    : _config(cfg),
      _context(context),
      _initialized(false),
      _luma_kernel_norm(nullptr),
      _luma_kernel_raw(nullptr),
      _layer_1_kernel(nullptr),
      _layer_2_kernel(nullptr),
      _layer_3_kernel(nullptr),
      _sum_sq_kernel(nullptr),
      _sum_kernel(nullptr),
      _subtract_from_all_kernel(nullptr) {}

void DataPipeline::init(int load_flags) {
  load_kernels(load_flags);
  _initialized = true;
}

void DataPipeline::load_kernels(int load_flags) {
  bool load_luma = (load_flags & DataPipeline::LOAD_KERNEL_LUMA) != 0,
       load_layers = (load_flags & DataPipeline::LOAD_KERNEL_LAYERS) != 0,
       load_misc = (load_flags & DataPipeline::LOAD_KERNEL_MISC) != 0;
  if (load_layers && !_config) {
    throw std::runtime_error(
        "Tried to load layer kernels without provided config");
  }

  /* clang-format off */
  if(load_luma && !_luma_kernel_norm)
    _luma_kernel_norm = _context->create_kernel(luma_kernel_file, "-D NORMALIZE");
  if(load_luma && !_luma_kernel_raw)
    _luma_kernel_raw = _context->create_kernel(luma_kernel_file);

  if(load_layers && !_layer_1_kernel)
    _layer_1_kernel = _layer_executor.create_layer_kernel(_context, layer_kernel_file, _config->n1);
  if(load_layers && !_layer_2_kernel)
    _layer_2_kernel = _layer_executor.create_layer_kernel(_context, layer_kernel_file, _config->n2);
  if(load_layers && !_layer_3_kernel)
    _layer_3_kernel = _layer_executor.create_layer_kernel(_context, layer_kernel_file, 1, 255);

  if(load_misc && !_sum_sq_kernel)
    _sum_sq_kernel = _context->create_kernel(sum_sq_kernel_file);
  if(load_misc && !_sum_kernel)
    _sum_kernel = _context->create_kernel(sum_kernel_file);
  if(load_misc && !_subtract_from_all_kernel)
    _subtract_from_all_kernel = _context->create_kernel(subtract_from_all_kernel_file);
  /* clang-format on */
}

void DataPipeline::check_initialized(int kernel_load_flags) {
  if (!_initialized) {
    throw std::runtime_error(
        "Tried to use DataPipeline before it was initialized");
  }

  this->load_kernels(kernel_load_flags);
}

cl_event DataPipeline::extract_luma(opencl::utils::ImageData &img_data,
                                    opencl::MemoryHandler *&gpu_buf_out,
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
  auto gpu_image = _context->create_image(
      CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  _context->write_image(gpu_image, img_data, true);
  gpu_buf_out =
      _context->allocate(CL_MEM_READ_WRITE, sizeof(cl_float) * out_pixel_count);

  // kernel args
  kernel->push_arg(gpu_image);
  kernel->push_arg(gpu_buf_out);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&img_data.h);

  // Launch kernel
  return kernel->execute(2, global_work_size, local_work_size, ev_to_wait_for,
                         ev_to_wait_for ? 1 : 0);
}

cl_event DataPipeline::execute_cnn(LayerData &layer_1, LayerData &layer_2,
                                   LayerData &layer_3,
                                   opencl::MemoryHandler *input,
                                   opencl::MemoryHandler *&gpu_buf_out,
                                   size_t upscaled_w, size_t upscaled_h,
                                   bool subtract_input_mean,
                                   cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_LAYERS);
  opencl::MemoryHandler *layer_1_out, *layer_2_out;
  size_t l2_input[2], l3_input[2];
  layer_1.get_output_dimensions(l2_input, upscaled_w, upscaled_h);
  layer_2.get_output_dimensions(l3_input, l2_input[0], l2_input[1]);

  _context->block();

  cl_event ev;
  if (subtract_input_mean) {
    std::cout << "### Subtracting mean from input" << std::endl;
    ev = this->subtract_mean(input, ev_to_wait_for);
    ev_to_wait_for = &ev;
  }

  _context->block();

  // layer 1
  std::cout << "### Executing layer 1" << std::endl;
  cl_event finish_token1 =
      _layer_executor(*_layer_1_kernel, layer_1, input, upscaled_w, upscaled_h,
                      layer_1_out, ev_to_wait_for);
  _context->block();

  // layer 2
  std::cout << "### Executing layer 2" << std::endl;
  cl_event finish_token2 =
      _layer_executor(*_layer_2_kernel, layer_2, layer_1_out,  //
                      l2_input[0], l2_input[1], layer_2_out, &finish_token1);
  _context->block();

  // layer 3
  std::cout << "### Executing layer 3" << std::endl;
  cl_event finish_token3 =
      _layer_executor(*_layer_3_kernel, layer_3, layer_2_out,  //
                      l3_input[0], l3_input[1], gpu_buf_out, &finish_token2);
  _context->block();

  return finish_token3;
}

float DataPipeline::mean_squared_error(  // TODO change return type to double
    opencl::MemoryHandler *gpu_buf_ground_truth,
    opencl::MemoryHandler *gpu_buf_algo_res, size_t ground_truth_w,
    size_t ground_truth_h, cl_event *ev_to_wait_for) {
  check_initialized(DataPipeline::LOAD_KERNEL_MISC);
  size_t wasted = _config->f1 + _config->f2 + _config->f3 - 3,
         algo_w = ground_truth_w - wasted, algo_h = ground_truth_h - wasted,
         algo_size = algo_w * algo_h;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(*_sum_sq_kernel, global_work_size, local_work_size,
                            algo_w, algo_h);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  const float out_init_val = 0;
  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  _context->write_buffer(gpu_buf_out, (void *)&out_init_val, true);  // zeroe

  // kernel args
  _sum_sq_kernel->push_arg(gpu_buf_ground_truth);
  _sum_sq_kernel->push_arg(gpu_buf_algo_res);
  _sum_sq_kernel->push_arg(sizeof(cl_float) * local_work_size[0],
                           nullptr);  // scrath
  _sum_sq_kernel->push_arg(gpu_buf_out);
  _sum_sq_kernel->push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  _sum_sq_kernel->push_arg(sizeof(cl_uint), (void *)&algo_w);
  _sum_sq_kernel->push_arg(sizeof(cl_uint), (void *)&algo_size);

  // run
  cl_event finish_token = _sum_sq_kernel->execute(
      1, global_work_size, local_work_size, ev_to_wait_for);

  // read (values may not be exactly the same since float->long data loss,
  // but should be close enough)
  float read_val;
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_ulong), (void *)&read_val,
                        true, &finish_token, 1);

  float mse_per_pixel = read_val / algo_size;
  std::cout << "Mean squared error raw: " << read_val
            << ", per pixel: " << mse_per_pixel << std::endl;
  return mse_per_pixel;
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
  auto gpu_buf_out = _context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  _context->write_buffer(gpu_buf_out, (void *)result, true);  // zeroe

  // kernel args
  _sum_kernel->push_arg(data);
  _sum_kernel->push_arg(gpu_buf_out);
  _sum_kernel->push_arg(sizeof(cl_float) * local_work_size[0],
                        nullptr);  // scrath
  _sum_kernel->push_arg(sizeof(cl_uint), (void *)&len);

  // run
  cl_event finish_token = _sum_kernel->execute(1, global_work_size,
                                               local_work_size, ev_to_wait_for);

  // read (values may not be exactly the same since float->long data loss,
  // but should be close enough)
  _context->read_buffer(gpu_buf_out, 0, sizeof(cl_ulong), (void *)result, true,
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

void DataPipeline::debug_buffer(opencl::MemoryHandler *data, size_t rows) {
  _context->block();
  size_t len = data->size / sizeof(cl_float), cols = len / rows;
  std::unique_ptr<float[]> cpu_buf(new float[len]);
  _context->read_buffer(data, (void *)cpu_buf.get(), true);

  for (size_t i = 0; i < rows; i++) {
    std::cout << "[ ";
    for (size_t j = 0; j < cols; j++) {
      std::cout << cpu_buf[cols * i + j] << ", ";
    }
    std::cout << "]" << std::endl;
  }
}
}
