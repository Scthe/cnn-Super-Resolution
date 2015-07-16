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
///
///
void print_buffer(opencl::Context& ctx, opencl::MemoryHandle mh,
                  const char* const name, size_t per_line) {
  auto raw = ctx.raw_memory(mh);
  size_t len = raw->size / sizeof(cl_float);
  // std::cout << "len:" << len << std::endl;

  // read
  std::vector<float> data(len);
  ctx.block();
  ctx.read_buffer(mh, &data[0], true);

  // print
  std::cout << name << ": [" << std::endl;
  for (size_t i = 0; i < len / per_line; i++) {
    std::cout << "[" << i << "] ";
    for (size_t j = 0; j < per_line; j++) {
      size_t idx = i * per_line + j;
      if (idx < len) std::cout << data[idx];
      if (j + 1 != per_line) std::cout << ", ";
    }
    if (i + 1 != len / per_line) std::cout << std::endl;
  }
  std::cout << "]" << std::endl
            << std::endl
            << std::endl;
}

///
///
///
void dump_image(const char* const file_path, size_t w,
                std::vector<float>& source, bool is_source_single_channel,
                float val_mul = 1.0f) {
  // TODO add resize - f.e. when viewing weights 9px x 9px is kind of small
  size_t channel_count = is_source_single_channel ? 1 : 3,
         h = source.size() / w / channel_count;
  std::cout << "dumping image(w=" << w << "x" << h << "x" << channel_count
            << ")to: '" << file_path << "'" << std::endl;
  std::vector<unsigned char> data(w * h * 3);
  for (size_t row = 0; row < h; row++) {
    for (size_t col = 0; col < w; col++) {
      size_t idx = (row * w + col) * channel_count;
      for (size_t k = 0; k < 3; k++) {
        float val = source[is_source_single_channel ? idx : idx + k] * val_mul;
        data[(row * w + col) * 3 + k] = (unsigned char)val;
      }
      /*
        */
    }
  }

  ImageData dd(w, h, sizeof(unsigned char) * 3, &data[0]);
  opencl::utils::write_image(file_path, dd);
}
/*
void dump_weights(const char* const path, cnn_sr::LayerData& data,
                  cnn_sr::CnnLayerGpuAllocationPool gpu_alloc) {
  // read
  auto ws = data.weights_size();
  std::vector<float> w(ws);
  context.read_buffer(gpu_alloc.weights, (void*)&w[0], true);
  // dump
  char name_buf[255];
  std::vector<float> tmp(data.f_spatial_size * data.f_spatial_size);
  for (size_t n = 0; n < data.current_filter_count; n++) {
    for (size_t k = 0; k < data.n_prev_filter_cnt; k++) {
      size_t i = 0;
      for (size_t row = 0; row < data.f_spatial_size; row++) {
        for (size_t col = 0; col < data.f_spatial_size; col++) {
          size_t idx = ((row * data.f_spatial_size) + col) *
                       data.current_filter_count * data.n_prev_filter_cnt;
          idx += k * data.current_filter_count + n;
          tmp[i] = ws[idx];
          ++i;
        }
      }

      snprintf(name_buf, 255, "%s__%d_%d.png", path, k, n);
      dump_image(name_buf, data.f_spatial_size, tmp, true, 255.0f);
    }
  }
}*/

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
    opencl::Context context;
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
    context.raw_memory(gpu_alloc.input_data)->release();  // not needed

    // std::cout << "--- expected_output_img debug ---" << std::endl;
    // print_buffer(context, gpu_alloc.expected_output_luma,
    //  "expected_output_luma", 16);
    // std::vector<float> luma_data(expected_output_img.w * expected_output_img.h);
    // context.read_buffer(gpu_alloc.expected_output_luma, (void*)&luma_data[0],
                        // true);
    // dump_image("data\\luma_extract.png", expected_output_img.w, luma_data, true,
              //  255.0f);
    // exit(EXIT_SUCCESS);

    // const size_t iters = 1000;
    const size_t iters = 990;
    for (size_t iter = 0; iter < iters; iter++) {
      //
      // process with layers
      auto finish_token3 = data_pipeline.forward(
          gpu_alloc.layer_1,  //
          gpu_alloc.layer_2,  //
          gpu_alloc.layer_3,  //
          gpu_alloc.input_luma, input_img.w, input_img.h, true);

      context.block();

      //
      // squared difference
      auto squared_error = data_pipeline.squared_error(
          gpu_alloc.input_luma,      //
          gpu_alloc.layer_3.output,  //
          expected_output_img.w, expected_output_img.h, &finish_token3);
      std::cout << "Squared error: " << squared_error
                << ", squared error per pixel: "
                << (squared_error / expected_output_img.w /
                    expected_output_img.h) << std::endl;
      // if (squared_error < 15) break;

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

      //
      // print buffers
      // print_buffer(context, gpu_alloc.layer_1.output, "layer 1 out", 14);
      // print_buffer(context, gpu_alloc.layer_2.output, "layer 2 out", 14);
      // print_buffer(context, gpu_alloc.layer_3.output, "layer 3 out", 12);
      // print_buffer(context, gpu_alloc.layer_3.deltas, "last layer deltas",
      // 12);

      // context.print_app_memory_usage();
    }
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
