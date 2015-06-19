#include <iostream>

#include "opencl\Context.hpp"
#include "opencl\Kernel.hpp"
#include "opencl\UtilsOpenCL.hpp"

#include "Config.hpp"
#include "LayerData.hpp"
// #include "LayerExecutor.hpp"
#include "DataPipeline.hpp"
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

///
///
///
int main(int argc, char **argv) {
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

    DataPipeline data_pipeline(&cfg, &context);
    data_pipeline.init();

    // load images
    // (small to process, large to mean square error)
    // TODO naive upscale for small image
    opencl::utils::ImageData img_large, img_small;
    opencl::MemoryHandler *luma_result_buf_large, *luma_result_buf_small;
    opencl::utils::load_image(img_large_file, img_large);
    std::cout << "img_large: " << img_large.w << "x" << img_large.h << "x"
              << img_large.bpp << std::endl;
    opencl::utils::load_image(img_small_file, img_small);
    std::cout << "img_small: " << img_small.w << "x" << img_small.h << "x"
              << img_small.bpp << std::endl;
    /* clang-format off */
    cl_event finish_token1 = data_pipeline.extract_luma(img_large, luma_result_buf_large, false);
    cl_event finish_token2 = data_pipeline.extract_luma(img_small, luma_result_buf_small, false, &finish_token1);
    /* clang-format on */

    // process with layers
    LayerData layer_1 = LayerData::from_N_distribution(1, cfg.n1, cfg.f1);
    LayerData layer_2 = LayerData::from_N_distribution(cfg.n1, cfg.n2, cfg.f2);
    LayerData layer_3 = LayerData::from_N_distribution(cfg.n2, 1, cfg.f3);
    opencl::MemoryHandler *cnn_out;
    auto finish_token3 = data_pipeline.execute_cnn(
        layer_1, layer_2, layer_3, luma_result_buf_small, cnn_out, img_small.w,
        img_small.h, &finish_token2);

    // mean square error
    // TODO luma is 0-1 or 0-255 ? (res:0-1)
    std::cout << "### Calcutating mean squared error" << std::endl;
    auto mse = data_pipeline.mean_squared_error(luma_result_buf_large, cnn_out,
                                                img_large.w, img_large.h,
                                                &finish_token3);

    std::cout << "mse: " << mse << std::endl;

    // TODO release MemoryHandlers !

  } catch (const std::exception &e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}
