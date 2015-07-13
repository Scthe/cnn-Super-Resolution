#include <iostream>

#include "Config.hpp"
#include "LayerData.hpp"
#include "ConfigBasedDataPipeline.hpp"
#include "Utils.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

using namespace opencl::utils;
using namespace cnn_sr;

struct GpuAllocationPool {
  /** Raw 3 channel image loaded from hard drive */
  opencl::MemoryHandle input_data = gpu_nullptr;
  /** Single channel (luma) of size input_img_w*input_img_h */
  opencl::MemoryHandle input_luma = gpu_nullptr;

  CnnLayerGpuAllocationPool layer_1;
  CnnLayerGpuAllocationPool layer_2;
  CnnLayerGpuAllocationPool layer_3;

  /** Raw 3 channel image loaded from disc */
  opencl::MemoryHandle expected_output_data = gpu_nullptr;
  /** Used only during training */
  opencl::MemoryHandle expected_output_luma = gpu_nullptr;
};

void prepare_image(DataPipeline* const pipeline, const char* const, ImageData&,
                   opencl::MemoryHandle&, opencl::MemoryHandle&, bool);

///
/// main
///
int main(int argc, char** argv) {
  const char* const cfg_file = "data\\config.json";
  const char* const input_img_file = "data\\small.jpg";
  const char* const expected_output_img_file = "data\\large.jpg";

  try {
    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;

    // opencl context
    opencl::Context context(argc, argv);
    context.init();

    ConfigBasedDataPipeline data_pipeline(cfg, &context);
    data_pipeline.init(/*DataPipeline::LOAD_KERNEL_NONE*/);
    GpuAllocationPool gpu_alloc;  // memory handles

    //
    // read & prepare images
    ImageData expected_output_img, input_img;
    prepare_image(
        &data_pipeline, expected_output_img_file, expected_output_img,  //
        gpu_alloc.expected_output_data, gpu_alloc.expected_output_luma, true);
    prepare_image(&data_pipeline, input_img_file, input_img,
                  gpu_alloc.input_data, gpu_alloc.input_luma, true);

    context.block();

    //
    // process with layers
    auto finish_token3 = data_pipeline.forward(gpu_alloc.layer_1,  //
                                               gpu_alloc.layer_2,  //
                                               gpu_alloc.layer_3,  //
                                               gpu_alloc.input_luma,
                                               input_img.w, input_img.h, true);

    context.block();

    //
    // squared difference
    // TODO luma is 0-1 or 0-255 ? (res:0-1)
    // std::cout << "### Calcutating mean squared error" << std::endl;
    // auto mse = data_pipeline.mean_squared_error(luma_result_buf_large,
    // cnn_out, img_large.w, img_large.h, &finish_token3);
    // std::cout << "error: " << mse << std::endl;

    // exit(EXIT_SUCCESS);

    //
    // backpropagate
    auto finish_token4 =
        data_pipeline.backpropagate(gpu_alloc.layer_1,               //
                                    gpu_alloc.layer_2,               //
                                    gpu_alloc.layer_3,               //
                                    gpu_alloc.input_luma,            //
                                    gpu_alloc.expected_output_luma,  //
                                    input_img.w,
                                    input_img.h,  //
                                    &finish_token3);

  } catch (const std::exception& e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}

void prepare_image(DataPipeline* const pipeline, const char* const file_path,
                   ImageData& img_data, opencl::MemoryHandle& gpu_data_handle,
                   opencl::MemoryHandle& gpu_luma_handle, bool normalize_luma) {
  std::cout << "loading image '" << file_path << "'" << std::endl;
  opencl::utils::load_image(file_path, img_data);  // TODO should throw
  std::cout << "    size: " << img_data.w << "x" << img_data.h << "x"
            << img_data.bpp << std::endl;
  // extract luma channel
  cl_event finish_token1 = pipeline->extract_luma(
      img_data, gpu_data_handle, gpu_luma_handle, normalize_luma);
}
