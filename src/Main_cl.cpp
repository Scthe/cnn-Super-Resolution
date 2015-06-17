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

    opencl::Context context(argc, argv);
    context.init();

    // load kernels
    auto luma_kernel = context.create_kernel(luma_kernel_file);
    auto layer_kernel = context.create_kernel(layer_kernel_file);
    auto sum_sq_kernel = context.create_kernel(sum_sq_kernel_file);

    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;

    // load images
    opencl::utils::ImageData img_large;
    opencl::utils::load_image(img_large_file, img_large);
    std::cout << "img_large: " << img_large.w << "x" << img_large.h << "x"
              << img_large.bpp << std::endl;
    opencl::utils::ImageData img_small;
    opencl::utils::load_image(img_small_file, img_small);
    std::cout << "img_small: " << img_small.w << "x" << img_small.h << "x"
              << img_small.bpp << std::endl;

    // read/init layer data
    LayerData layer_1 = LayerData::from_N_distribution(1, cfg.n1, cfg.f1);
    LayerData layer_2 = LayerData::from_N_distribution(cfg.n1, cfg.n2, cfg.f2);
    LayerData layer_3 = LayerData::from_N_distribution(cfg.n2, 1, cfg.f3);

    // TODO naive upscale for small image

    // extract luma from small image
    // (small to process, large to mean square error)
    opencl::MemoryHandler *luma_result_buf_small, *luma_result_buf_large;
    auto finish_token = extract_luma(*luma_kernel, img_small, luma_result_buf_small);
    finish_token = extract_luma(*luma_kernel, img_large, luma_result_buf_large, &finish_token);

    // process with layers
    // LayerExecutor layer_executor;
    // layer_executor

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
  auto gpu_image = context->create_image(CL_MEM_READ_WRITE, CL_RGBA, CL_UNSIGNED_INT8, img_data.w, img_data.h);
  context->write_image(gpu_image, img_data, true);
  gpu_buf_out = context->allocate(CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * out_pixel_count);
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

// TODO remove all code below - it is just as code sample now

///
///
///

void cfg_tests() {
  using namespace cnn_sr;
  ConfigReader reader;
  // reader.read("non_exists.json");
  // reader.read("tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___.json");
  // reader.read("data\\config_err.json");
  Config cfg = reader.read("data\\config.json");

  std::cout << cfg << std::endl;
}

///
///
///

void layerData_tests() {
  using namespace cnn_sr;
  std::vector<LayerData> data;
  data.push_back(LayerData(1, 3, 3));
  data.push_back(LayerData(3, 2, 3));
  data.push_back(LayerData(3, 3, 1));
  data.push_back(LayerData(3, 1, 3));

  // read from file
  const char *const layer_keys[4] = {"layer_1", "layer_2_data_set_1",
                                     "layer_2_data_set_2", "layer_3"};

  LayerParametersIO param_reader;
  std::cout << "reading" << std::endl;
  param_reader.read("data/layer_data_example.json", data, layer_keys,
                    NUM_ELEMS(layer_keys));

  for (auto a : data) {
    std::cout << a << std::endl;
  }

  std::cout << "from_N_distribution" << std::endl;
  LayerData nn = LayerData::from_N_distribution(1, 3, 3);
}

///
///
///

void LayerExecutor_tests() {
  using namespace cnn_sr;
  opencl::Kernel kernel;

  // LayerExecutor exec;
  /*
  size_t gws[2];
  size_t lws[2];
  exec.work_sizes(kernel, gws, lws, 16523, 5);
  std::cout << "global: " << gws[0] << ", " << gws[1] << std::endl;
  std::cout << "local: " << lws[0] << ", " << lws[1] << std::endl;
  std::cout << "(global should be closes power2)" << std::endl;
  std::cout << "(local[0]*local[1] should be < CONST)" << std::endl;
  */
}

///
///
///

#define OUT_CHANNELS 1

void luma_extract(int argc, char **argv) {
  using namespace opencl::utils;

  const char *cSourceFile = "src/kernel/greyscale.cl";
  const char *img_path = "data/cold_reflection_by_lildream.jpg";
  const char *img_path2 = "data/cold_2.png";

  ImageData data;
  load_image(img_path, data);
  std::cout << "img: " << data.w << "x" << data.h << "x" << data.bpp
            << std::endl;

  size_t global_work_size[2] = {512, 512};  // ceil
  size_t local_work_size[2] = {32, 32};

  //
  // opencl context
  opencl::Context context(argc, argv);
  context.init();

  // memory allocation - both CPU & GPU
  size_t pixel_total = data.w * data.h,  //
      data_total = pixel_total * OUT_CHANNELS;

  auto gpu_image = context.create_image(CL_MEM_READ_WRITE, CL_RGBA,
                                        CL_UNSIGNED_INT8, data.w, data.h);
  context.write_image(gpu_image, data, true);

  auto gpu_buf =
      context.allocate(CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * data_total);
  std::cout << "cpu/gpu buffers pair allocated" << std::endl;
  std::cout << "kernel create" << std::endl;

  auto kernel = context.create_kernel(cSourceFile);

  std::cout << "push args" << std::endl;

  // kernel args
  kernel->push_arg(sizeof(cl_mem), (void *)&gpu_image->handle);
  kernel->push_arg(sizeof(cl_mem), (void *)&gpu_buf->handle);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.w);
  kernel->push_arg(sizeof(cl_uint), (void *)&data.h);

  std::cout << "execute" << std::endl;

  // Launch kernel
  cl_event finish_token = kernel->execute(2, global_work_size, local_work_size);

  // Synchronous/blocking read of results
  std::cout << "create result structure" << std::endl;
  ImageData out;
  out.w = data.w;
  out.h = data.h;
  out.bpp = OUT_CHANNELS;
  out.data = new unsigned char[data_total];
  std::cout << "read result" << std::endl;
  context.read_buffer(gpu_buf,                                             //
                      0, sizeof(cl_uchar) * data_total, (void *)out.data,  //
                      true, &finish_token, 1);

  /*
  int base_row = 90, base_col = 85;
  for (size_t i = 0; i < 2; i++) {
    int base_r = (i + base_row) * 500;
    for (size_t j = 0; j < 32; j++) {
      size_t idx = base_r + base_col + j;
      std::cout << (int)out.data[idx] << " ";
    }
    std::cout << std::endl;
    for (size_t j = 0; j < 32; j++) {
      size_t idx = base_r + base_col + j;
      std::cout << (int)data.data[idx * 4 + 2] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
  */

  std::cout << std::endl
            << "--write--" << std::endl;
  int res = opencl::utils::write_image(img_path2, out);

  std::cout << "write status: " << res << std::endl
            << "--end--" << std::endl;
}
