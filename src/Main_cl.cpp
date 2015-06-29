#include <iostream>

#include "Config.hpp"
#include "LayerData.hpp"
#include "DataPipeline.hpp"
#include "Utils.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

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

struct GpuAllocationPool {
  /**
   * Single channel (luma) of size:
   *     input_img_w * input_img_h
   */
  opencl::MemoryHandler* cnn_input = nullptr;
  /** Raw 3 channel image loaded from hard drive */
  opencl::MemoryHandler* cnn_input_raw = nullptr;

  cnn_sr::CnnLayerGpuAllocationPool layer_1_pool;
  cnn_sr::CnnLayerGpuAllocationPool layer_2_pool;
  cnn_sr::CnnLayerGpuAllocationPool layer_3_pool;

  /** Used only during training */
  opencl::MemoryHandler* cnn_expected_output = nullptr;
  /** Raw 3 channel image loaded from disc. Used only during training */
  opencl::MemoryHandler* cnn_expected_output_raw = nullptr;

  /** Used only during training */
  opencl::MemoryHandler* mean_squared_error = nullptr;
};

///
/// main
///
int main(int argc, char** argv) {
  const char* const cfg_file = "data\\config.json";
  const char* const img_small_file = "data\\small.jpg";
  const char* const img_large_file = "data\\large.jpg";

  try {
    using namespace cnn_sr;
    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;

    // opencl context
    opencl::Context context(argc, argv);
    context.init();

    DataPipeline data_pipeline(&cfg, &context);
    data_pipeline.init();

    GpuAllocationPool gpu_alloc;

    // load images
    // (small to process, large to mean square error)
    opencl::utils::ImageData img_large, img_small;
    opencl::utils::load_image(img_large_file, img_large);
    std::cout << "img_large: " << img_large.w << "x" << img_large.h << "x"
              << img_large.bpp << std::endl;
    opencl::utils::load_image(img_small_file, img_small);
    std::cout << "img_small: " << img_small.w << "x" << img_small.h << "x"
              << img_small.bpp << std::endl;
    // load images into gpu, preprocess
    /* clang-format off */
    cl_event finish_token1 = data_pipeline.extract_luma(img_large,
                                   gpu_alloc.cnn_expected_output_raw,
                                   gpu_alloc.cnn_expected_output, false);
    cl_event finish_token2 = data_pipeline.extract_luma(img_small,
                                   gpu_alloc.cnn_input_raw,
                                   gpu_alloc.cnn_input, true, &finish_token1);
    /* clang-format on */

    // process with layers
    LayerData layer_1 = LayerData::from_N_distribution(1, cfg.n1, cfg.f1);
    LayerData layer_2 = LayerData::from_N_distribution(cfg.n1, cfg.n2, cfg.f2);
    LayerData layer_3 = LayerData::from_N_distribution(cfg.n2, 1, cfg.f3);

    auto finish_token3 = data_pipeline.execute_cnn(
        layer_1, gpu_alloc.layer_1_pool,  //
        layer_2, gpu_alloc.layer_2_pool,  //
        layer_3, gpu_alloc.layer_3_pool,  //
        gpu_alloc.cnn_input, img_small.w, img_small.h, true, &finish_token2);

    // mean square error
    // TODO luma is 0-1 or 0-255 ? (res:0-1)
    // std::cout << "### Calcutating mean squared error" << std::endl;
    // auto mse = data_pipeline.mean_squared_error(luma_result_buf_large,
    // cnn_out, img_large.w, img_large.h, &finish_token3);
    // std::cout << "mse: " << mse << std::endl;
  } catch (const std::exception& e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}
