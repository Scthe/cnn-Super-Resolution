#include <iostream>
#include <algorithm>  // for random_shuffle
#include <stdexcept>  // for runtime_exception
#include <ctime>      // random seed

#include "Config.hpp"
#include "LayerData.hpp"
#include "ConfigBasedDataPipeline.hpp"
#include "Utils.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

using namespace opencl::utils;
using namespace cnn_sr;

// SNIPPET -- PARAMETER WRITE:
// context.block();
// data_pipeline.write_params_to_file("data\\params-file.json",
//  gpu_alloc.layer_1, gpu_alloc.layer_2,
//  gpu_alloc.layer_3);
// exit(EXIT_SUCCESS);

// SNIPPET -- LUMA WRITE:
// std::cout << "--- expected_output_img debug ---" << std::endl;
// data_pipeline.print_buffer(gpu_alloc.expected_output_luma,
//  "expected_output_luma", 16);
// std::vector<float> luma_data(expected_output_img.w *
// expected_output_img.h);
// context.read_buffer(gpu_alloc.expected_output_luma, (void*)&luma_data[0],
// true);
// dump_image("data\\luma_extract.png", expected_output_img.w, luma_data,
// true,
//  255.0f);
// exit(EXIT_SUCCESS);

// const char* const input_img_file = "data\\small.jpg";
// const char* const expected_output_img_file = "data\\large.jpg";
const char* train_samples[24] = {"data\\train_samples\\sample_0_large.jpg",
                                 "data\\train_samples\\sample_0_small.jpg",
                                 "data\\train_samples\\sample_1_large.jpg",
                                 "data\\train_samples\\sample_1_small.jpg",
                                 "data\\train_samples\\sample_2_large.jpg",
                                 "data\\train_samples\\sample_2_small.jpg",
                                 "data\\train_samples\\sample_3_large.jpg",
                                 "data\\train_samples\\sample_3_small.jpg",
                                 "data\\train_samples\\sample_4_large.jpg",
                                 "data\\train_samples\\sample_4_small.jpg",
                                 "data\\train_samples\\sample_5_large.jpg",
                                 "data\\train_samples\\sample_5_small.jpg",
                                 "data\\train_samples\\sample_7_large.jpg",
                                 "data\\train_samples\\sample_7_small.jpg",
                                 "data\\train_samples\\sample_8_large.jpg",
                                 "data\\train_samples\\sample_8_small.jpg",
                                 "data\\train_samples\\sample_9_large.jpg",
                                 "data\\train_samples\\sample_9_small.jpg",
                                 "data\\train_samples\\sample_10_large.jpg",
                                 "data\\train_samples\\sample_10_small.jpg",
                                 "data\\train_samples\\sample_11_large.jpg",
                                 "data\\train_samples\\sample_11_small.jpg",
                                 "data\\train_samples\\sample_12_large.jpg",
                                 "data\\train_samples\\sample_12_small.jpg"};
const size_t train_samples_count = 12;

struct PerSampleAllocationPool {
  /** Raw 3 channel image loaded from hard drive */
  opencl::MemoryHandle input_data = gpu_nullptr;
  /** Single channel (luma) of size input_img_w*input_img_h */
  opencl::MemoryHandle input_luma = gpu_nullptr;
  /** Raw 3 channel image loaded from disc */
  opencl::MemoryHandle expected_output_data = gpu_nullptr;
  /** Used only during training */
  opencl::MemoryHandle expected_output_luma = gpu_nullptr;

  size_t w, h;
};

struct GpuAllocationPool {
  CnnLayerGpuAllocationPool layer_1;
  CnnLayerGpuAllocationPool layer_2;
  CnnLayerGpuAllocationPool layer_3;

  std::vector<PerSampleAllocationPool> samples;
};

///
/// Forward decl.
///
cl_event prepare_image(DataPipeline* const pipeline, const char* const,
                       ImageData&, opencl::MemoryHandle&, opencl::MemoryHandle&,
                       bool, bool);
void dump_image(const char* const, size_t w, std::vector<float>&, bool,
                float val_mul = 1.0f);

void divide_samples(size_t validation_set_size, GpuAllocationPool&,
                    std::vector<PerSampleAllocationPool>& train_set,
                    std::vector<PerSampleAllocationPool>& validation_set);

///
/// main
///
int main(int argc, char** argv) {
  std::srand(std::time(0));
  const char* const cfg_file = "data\\config_small.json";
  try {
    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;
    const size_t batches_count = 100;
    const size_t validation_set_size = 3;

    if (validation_set_size >= train_samples_count)
      throw std::runtime_error(
          "Provide more training samples or decrease validation set size");

    // opencl context
    opencl::Context context;
    context.init();
    ConfigBasedDataPipeline data_pipeline(cfg, &context);
    data_pipeline.init();
    GpuAllocationPool gpu_alloc;

    //
    // read & prepare images
    for (size_t i = 0; i < train_samples_count; i++) {
      ImageData expected_output_img, input_img;
      PerSampleAllocationPool sample_alloc_pool;
      auto large_path = train_samples[2 * i],
           small_path = train_samples[2 * i + 1];
      prepare_image(&data_pipeline, large_path, expected_output_img,
                    sample_alloc_pool.expected_output_data,
                    sample_alloc_pool.expected_output_luma, true, false);
      auto ev1 = prepare_image(&data_pipeline, small_path, input_img,
                               sample_alloc_pool.input_data,
                               sample_alloc_pool.input_luma, true, false);
      data_pipeline.subtract_mean(sample_alloc_pool.input_luma, &ev1);
      sample_alloc_pool.w = (size_t)input_img.w;
      sample_alloc_pool.h = (size_t)input_img.h;
      context.block();
      context.raw_memory(sample_alloc_pool.input_data)->release();
      gpu_alloc.samples.push_back(sample_alloc_pool);
    }

    context.block();

    // train
    std::vector<PerSampleAllocationPool> train_set(train_samples_count),
        validation_set(train_samples_count);
    for (size_t batch_id = 0; batch_id < batches_count; batch_id++) {
      // std::cout << "------ BATCH " << batch_id << " ------" << std::endl;
      float train_squared_error = 0;
      size_t train_px = 0;

      divide_samples(validation_set_size, gpu_alloc, train_set, validation_set);

      for (PerSampleAllocationPool& sample_alloc_pool : train_set) {
        const size_t w = sample_alloc_pool.w, h = sample_alloc_pool.h;
        train_px += w * h;

        // process with layers
        auto forward_ev =
            data_pipeline.forward(gpu_alloc.layer_1,  //
                                  gpu_alloc.layer_2,  //
                                  gpu_alloc.layer_3,  //
                                  sample_alloc_pool.input_luma, w, h);
        // squared difference
        train_squared_error +=
            data_pipeline.squared_error(sample_alloc_pool.input_luma,  //
                                        gpu_alloc.layer_3.output,      //
                                        w, h, &forward_ev);
        // backpropagate
        auto weight_decay_value = data_pipeline.weight_decay(
            gpu_alloc.layer_1, gpu_alloc.layer_2, gpu_alloc.layer_3,
            cfg.weight_decay_parameter, &forward_ev);
        auto finish_token4 = data_pipeline.backpropagate(
            gpu_alloc.layer_1,                       //
            gpu_alloc.layer_2,                       //
            gpu_alloc.layer_3,                       //
            sample_alloc_pool.input_luma,            //
            sample_alloc_pool.expected_output_luma,  //
            w, h, weight_decay_value);

        context.block();
      }

      // update_parameters
      data_pipeline.update_parameters(gpu_alloc.layer_1, gpu_alloc.layer_2,
                                      gpu_alloc.layer_3, batches_count);
      context.zeros_float(gpu_alloc.layer_1.accumulating_grad_w, true);
      context.zeros_float(gpu_alloc.layer_2.accumulating_grad_w, true);
      context.zeros_float(gpu_alloc.layer_3.accumulating_grad_w, true);
      context.zeros_float(gpu_alloc.layer_1.accumulating_grad_b, true);
      context.zeros_float(gpu_alloc.layer_2.accumulating_grad_b, true);
      context.zeros_float(gpu_alloc.layer_3.accumulating_grad_b, true);

      float validation_squared_error = 0.0f;
      size_t valid_px = 0;
      for (PerSampleAllocationPool& sample_alloc_pool : validation_set) {
        const size_t w = sample_alloc_pool.w, h = sample_alloc_pool.h;
        valid_px += w * h;

        // process with layers
        auto forward_ev =
            data_pipeline.forward(gpu_alloc.layer_1,  //
                                  gpu_alloc.layer_2,  //
                                  gpu_alloc.layer_3,  //
                                  sample_alloc_pool.input_luma, w, h);
        // squared difference
        validation_squared_error +=
            data_pipeline.squared_error(sample_alloc_pool.input_luma,  //
                                        gpu_alloc.layer_3.output,      //
                                        w, h, &forward_ev);
      }

      // (we are printing per pixel values because they are easier to remember)
      float mean_train_err = train_squared_error / train_set.size(),
            mean_valid_err = validation_squared_error / validation_set.size();
      std::cout << "[" << batch_id << "] "  //
                /*<< "mean train set error: " << mean_train_err << " ("
                << (mean_train_err / train_px) << " per px), "*/
                << "mean validation set error: " << mean_valid_err << " ("
                << (mean_valid_err / valid_px) << " per px)" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}

///
///
/// Impl
///
cl_event prepare_image(DataPipeline* const pipeline,
                       const char* const file_path, ImageData& img_data,
                       opencl::MemoryHandle& gpu_data_handle,
                       opencl::MemoryHandle& gpu_luma_handle,
                       bool normalize_luma, bool print) {
  if (print) std::cout << "loading image '" << file_path << "'" << std::endl;
  opencl::utils::load_image(file_path, img_data);  // TODO should throw
  if (print)
    std::cout << "    size: " << img_data.w << "x" << img_data.h << "x"
              << img_data.bpp << std::endl;
  // extract luma channel
  return pipeline->extract_luma(img_data, gpu_data_handle, gpu_luma_handle,
                                normalize_luma);
}

void dump_image(const char* const file_path, size_t w,
                std::vector<float>& source, bool is_source_single_channel,
                float val_mul) {
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
    }
  }

  ImageData dd(w, h, sizeof(unsigned char) * 3, &data[0]);
  opencl::utils::write_image(file_path, dd);
}

void divide_samples(size_t validation_set_size, GpuAllocationPool& pool,
                    std::vector<PerSampleAllocationPool>& train_set,
                    std::vector<PerSampleAllocationPool>& validation_set) {
  auto samples = pool.samples;
  train_set.clear();
  validation_set.clear();
  std::random_shuffle(samples.begin(), samples.end());
  auto st = samples.cbegin(), ne = std::next(st, validation_set_size);
  std::copy(st, ne, back_inserter(validation_set));
  std::copy(ne, samples.cend(), back_inserter(train_set));
}

/* clang-format off */
// data_pipeline.print_buffer(gpu_alloc.layer_1.bias, "layer 1 bias", 1);
// data_pipeline.print_buffer(gpu_alloc.layer_2.bias, "layer 2 bias", 1);
// data_pipeline.print_buffer(gpu_alloc.layer_3.bias, "layer 3 bias", 1);

// data_pipeline.print_buffer(gpu_alloc.layer_1.weights, "layer 1 weights", cfg.f1*cfg.f1);
// data_pipeline.print_buffer(gpu_alloc.layer_2.weights, "layer 2 weights", cfg.f2*cfg.f2);
// data_pipeline.print_buffer(gpu_alloc.layer_3.weights, "layer 3 weights", cfg.f3*cfg.f3);

// data_pipeline.print_buffer(gpu_alloc.layer_1.grad_w, "layer 1 weight gradients", cfg.f1*cfg.f1);
// data_pipeline.print_buffer(gpu_alloc.layer_2.grad_w, "layer 2 weight gradients", cfg.f2*cfg.f2);
// data_pipeline.print_buffer(gpu_alloc.layer_3.grad_w, "layer 3 weight gradients", cfg.f3*cfg.f3);

// data_pipeline.print_buffer(gpu_alloc.layer_1.output, "layer 1 out", l1_out_rows);
// data_pipeline.print_buffer(gpu_alloc.layer_2.output, "layer 2 out", l2_out_rows);
// data_pipeline.print_buffer(gpu_alloc.layer_3.output, "layer 3 out", l3_out_rows);

// data_pipeline.print_buffer(gpu_alloc.layer_1.deltas, "layer 1 deltas", l1_out_rows);
// data_pipeline.print_buffer(gpu_alloc.layer_2.deltas, "layer 2 deltas", l2_out_rows);
// data_pipeline.print_buffer(gpu_alloc.layer_3.deltas, "layer 3 deltas", l3_out_rows);
/* clang-format on */
