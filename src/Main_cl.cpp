#include <iostream>

#include "opencl\Context.hpp"
#include "opencl\Kernel.hpp"
#include "opencl\UtilsOpenCL.hpp"

#include "Config.hpp"
#include "LayerData.hpp"
#include "LayerExecutor.hpp"
#include "Utils.hpp"

/*
 * http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
 * http://www.thebigblob.com/gaussian-blur-using-opencl-and-the-built-in-images-textures/
 *
 * 1. load small.jpg, large.jpg
 * 2. create upscaled from small
 * 3. Extract luma from (2) and large
 * 4. go with the pipeline for (3)
 * 5. cmp. results with mean
 * 6. BACKPROPAGATE
 *
 * Later:
 * only patches from images
 *
 */

cl_event extract_luma(opencl::Kernel &, opencl::utils::ImageData &,
                      opencl::MemoryHandler *&, cl_event *ev = nullptr);

unsigned __int64
mean_squared_error(opencl::Kernel &kernel, cnn_sr::Config &cfg,
                   opencl::MemoryHandler *&gpu_buf_ground_truth,
                   opencl::MemoryHandler *&gpu_buf_algo_res,
                   size_t ground_truth_w, size_t ground_truth_h,
                   cl_event *finish_token);

///
///
///
int main(int argc, char **argv) {
  const char *const luma_kernel_file = "src/kernel/extract_luma.cl";
  const char *const layer_kernel_file = "src/kernel/layer_uber_kernel.cl";
  const char *const sum_sq_kernel_file = "src/kernel/sum_squared.cl";

  const char *const cfg_file = "data\\config.json";
  const char *const img_small_file = "data\\small.jpg";
  const char *const img_large_file = "data\\large.jpg";

  try {
    using namespace cnn_sr;

    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;

    // opencl context
    opencl::Context context(argc, argv);
    context.init();
    cl_event finish_token;

    // load kernels
    /* clang-format off */
    LayerExecutor layer_executor;
    auto luma_kernel_small = context.create_kernel(luma_kernel_file, "-D NORMALIZE");
    auto luma_kernel_large = context.create_kernel(luma_kernel_file);
    auto layer_1_kernel = layer_executor.create_layer_kernel(&context, layer_kernel_file, cfg.n1);
    auto layer_2_kernel = layer_executor.create_layer_kernel(&context, layer_kernel_file, cfg.n2);
    auto layer_3_kernel = layer_executor.create_layer_kernel(&context, layer_kernel_file, 1, 255);
    auto sum_sq_kernel = context.create_kernel(sum_sq_kernel_file);
    /* clang-format on */

    // load images - large
    opencl::utils::ImageData img_large;
    opencl::utils::load_image(img_large_file, img_large);
    std::cout << "img_large: " << img_large.w << "x" << img_large.h << "x"
              << img_large.bpp << std::endl;
    // load images - small
    opencl::utils::ImageData img_small;
    opencl::utils::load_image(img_small_file, img_small);
    std::cout << "img_small: " << img_small.w << "x" << img_small.h << "x"
              << img_small.bpp << std::endl;

    // TODO naive upscale for small image

    // extract luma
    // (small to process, large to mean square error)
    /* clang-format off */
    opencl::MemoryHandler *luma_result_buf_small, *luma_result_buf_large;
    finish_token = extract_luma(*luma_kernel_small, img_small, luma_result_buf_small);
    finish_token = extract_luma(*luma_kernel_large, img_large, luma_result_buf_large, &finish_token);
    /* clang-format on */

    // process with layers
    LayerData layer_1 = LayerData::from_N_distribution(1, cfg.n1, cfg.f1);
    LayerData layer_2 = LayerData::from_N_distribution(cfg.n1, cfg.n2, cfg.f2);
    LayerData layer_3 = LayerData::from_N_distribution(cfg.n2, 1, cfg.f3);
    opencl::MemoryHandler *layer_1_out, *layer_2_out, *layer_3_out;

    // layer 1
    context.block();
    std::cout << "### Executing layer 1" << std::endl;
    // TODO mean subtract
    finish_token =
        layer_executor(*layer_1_kernel, layer_1, luma_result_buf_small,
                       img_small.w, img_small.h, layer_1_out, &finish_token);
    context.block();

    // layer 2
    size_t l2_input_w = img_small.w - cfg.f1 + 1,
           l2_input_h = img_small.h - cfg.f1 + 1;
    std::cout << "### Executing layer 2" << std::endl;
    finish_token =
        layer_executor(*layer_2_kernel, layer_2, layer_1_out, l2_input_w,
                       l2_input_h, layer_2_out, &finish_token);
    context.block();

    // layer 3
    std::cout << "### Executing layer 3" << std::endl;
    size_t l3_input_w = l2_input_w - cfg.f2 + 1,
           l3_input_h = l2_input_h - cfg.f2 + 1;
    finish_token =
        layer_executor(*layer_3_kernel, layer_3, layer_2_out, l3_input_w,
                       l3_input_h, layer_3_out, &finish_token);
    context.block();

    // mean square error
    // TODO luma is 0-1 or 0-255 ? (res:0-1)
    std::cout << "### Calcutating mean squared error" << std::endl;
    auto mse = mean_squared_error(*sum_sq_kernel, cfg, luma_result_buf_large,
                                  layer_3_out, img_large.w, img_large.h,
                                  &finish_token);

    std::cout << "mse: " << mse << std::endl;

    // TODO release MemoryHandlers !

  } catch (const std::exception &e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}

///
///
///
cl_event extract_luma(opencl::Kernel &kernel,
                      opencl::utils::ImageData &img_data,
                      opencl::MemoryHandler *&gpu_buf_out, cl_event *ev) {
  opencl::Context *const context = kernel.get_context();
  size_t out_pixel_count = img_data.w * img_data.h;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size,
                            img_data.w, img_data.h);
  std::cout << "global work size: " << global_work_size[0] << ", "
            << global_work_size[1] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << ", "
            << local_work_size[1] << std::endl;

  // memory allocation
  auto gpu_image = context->create_image(
      CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  context->write_image(gpu_image, img_data, true);
  gpu_buf_out =
      context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_float) * out_pixel_count);
  // std::cout << "cpu/gpu buffers pair allocated" << std::endl;

  // std::cout << "push args" << std::endl;
  // kernel args
  kernel.push_arg(gpu_image);
  kernel.push_arg(gpu_buf_out);
  kernel.push_arg(sizeof(cl_uint), (void *)&img_data.w);
  kernel.push_arg(sizeof(cl_uint), (void *)&img_data.h);

  // Launch kernel
  // std::cout << "execute" << std::endl;
  return kernel.execute(2, global_work_size, local_work_size, ev, 1);
}

unsigned __int64
mean_squared_error(opencl::Kernel &kernel, cnn_sr::Config &cfg,
                   opencl::MemoryHandler *&gpu_buf_ground_truth,
                   opencl::MemoryHandler *&gpu_buf_algo_res,
                   size_t ground_truth_w, size_t ground_truth_h, cl_event *ev) {
  opencl::Context *const context = kernel.get_context();
  size_t wasted = cfg.f1 + cfg.f2 + cfg.f3 + 3,
         algo_w = ground_truth_w - wasted, algo_h = ground_truth_h - wasted,
         algo_size = algo_w * algo_h;

  size_t global_work_size[2];
  size_t local_work_size[2];
  opencl::utils::work_sizes(kernel, global_work_size, local_work_size, algo_w,
                            algo_h);
  global_work_size[0] *= global_work_size[1];
  local_work_size[0] *= local_work_size[1];
  std::cout << "global work size: " << global_work_size[0] << std::endl;
  std::cout << "local work size: " << local_work_size[0] << std::endl;

  const unsigned __int64 out_init_val = 0;
  auto gpu_buf_out = context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_ulong));
  context->write_buffer(gpu_buf_out, (void *)&out_init_val, true);  // zeroe

  // kernel args
  kernel.push_arg(gpu_buf_ground_truth);
  kernel.push_arg(gpu_buf_algo_res);
  kernel.push_arg(sizeof(cl_float) * local_work_size[0], nullptr);  // scrath
  kernel.push_arg(gpu_buf_out);
  kernel.push_arg(sizeof(cl_uint), (void *)&ground_truth_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&algo_w);
  kernel.push_arg(sizeof(cl_uint), (void *)&algo_size);

  // run
  cl_event finish_token =
      kernel.execute(1, global_work_size, local_work_size, ev);

  // read (values may not be exactly the same since float->long data loss,
  // but should be close enough)
  unsigned __int64 read_val;
  context->read_buffer(gpu_buf_out, 0, sizeof(cl_ulong), (void *)&read_val,
                       true, &finish_token, 1);

  return read_val;
}
