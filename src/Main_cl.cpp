#include <iostream>
#include <algorithm>  // for random_shuffle
#include <stdexcept>  // for runtime_exception
#include <ctime>      // random seed
#include <utility>    // for std::pair
#include <cmath>      // for std::isnan
#include <unordered_map>
#
#include "Config.hpp"
#include "LayerData.hpp"
#include "ConfigBasedDataPipeline.hpp"
#include "Utils.hpp"
#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

using namespace opencl::utils;
using namespace cnn_sr;

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

typedef std::pair<std::string, std::string> TrainSampleFiles;

void get_training_samples(std::string, std::vector<TrainSampleFiles>&);

void execute_app(opencl::Context& context, Config& cfg,
                 ConfigBasedDataPipeline& data_pipeline,
                 GpuAllocationPool& gpu_alloc) {
  // auto in_path = "data\\small.jpg";
  auto in_path = "data\\small2.jpg";
  auto out_path = "result.png";

  // read input image
  ImageData input_img;
  opencl::MemoryHandle input_data = gpu_nullptr;
  opencl::MemoryHandle input_luma = gpu_nullptr;
  auto ev1 = prepare_image(&data_pipeline, in_path,  //
                           input_img, input_data,    //
                           input_luma, true, false);
  data_pipeline.subtract_mean(input_luma, &ev1);
  size_t w = input_img.w, h = input_img.h,  //
      luma_w = w - cfg.total_padding(), luma_h = h - cfg.total_padding();
  context.block();

  // process with layers
  auto forward_ev = data_pipeline.forward(gpu_alloc.layer_1,  //
                                          gpu_alloc.layer_2,  //
                                          gpu_alloc.layer_3,  //
                                          input_luma, w, h);
  auto squared_error = data_pipeline.squared_error(input_luma,                //
                                                   gpu_alloc.layer_3.output,  //
                                                   w, h, &forward_ev);
  std::cout << "Squared error: " << squared_error << " ("
            << (squared_error / (luma_w * luma_h)) << " per px)" << std::endl;

  // dbg output read
  // data_pipeline.print_buffer(gpu_alloc.layer_1.output, "layer 1", h);
  data_pipeline.print_buffer(gpu_alloc.layer_2.output, "layer 2", h);
  // data_pipeline.print_buffer(gpu_alloc.layer_3.output, "OUT", h);

  // read image
  size_t result_size = w * h * 3;  // 3 channels
  opencl::MemoryHandle gpu_buf_target = gpu_nullptr;
  data_pipeline.swap_luma(input_img, input_data, gpu_alloc.layer_3.output,
                          gpu_buf_target, luma_w, luma_h);
  std::vector<unsigned char> result(result_size);
  context.read_buffer(gpu_buf_target, (void*)&result[0], true);

  opencl::utils::ImageData res_img(w, h, 3, &result[0]);
  opencl::utils::write_image(out_path, res_img);
}

///
/// main
///
int main(int argc, char** argv) {
  std::srand(std::time(0));
  // const char* const cfg_file = "data\\config_small.json";
  // const char* const cfg_file = "data\\config.json";
  // const char* const cfg_file = "data\\config_f.json";
  // bool train = true;
  bool train = false;
  const char* const cfg_file =
      train ? "data\\config.json" : "data\\config_f.json";
  const char* const train_samples_dir = "data\\train_samples";
  const char* const success_params_file = "data\\success_params_file.json";
  const size_t batches_count = 100;
  const size_t validation_set_size = 3;  // TODO use percentage

  std::vector<TrainSampleFiles> train_sample_files;
  get_training_samples(train_samples_dir, train_sample_files);
  if (validation_set_size >= train_sample_files.size()) {
    throw std::runtime_error(
        "Provide more training samples or decrease validation set size");
  }

  try {
    // read config
    ConfigReader reader;
    Config cfg = reader.read(cfg_file);
    std::cout << cfg << std::endl;

    // opencl context
    opencl::Context context;
    context.init();
    ConfigBasedDataPipeline data_pipeline(cfg, &context);
    data_pipeline.init();
    GpuAllocationPool gpu_alloc;

    if (!train) {
      execute_app(context, cfg, data_pipeline, gpu_alloc);
      exit(EXIT_SUCCESS);
    }

    //
    // read & prepare images
    for (auto& path_pair : train_sample_files) {
      ImageData expected_output_img, input_img;
      PerSampleAllocationPool sample_alloc_pool;
      prepare_image(&data_pipeline, path_pair.first.c_str(),
                    expected_output_img, sample_alloc_pool.expected_output_data,
                    sample_alloc_pool.expected_output_luma, true, false);
      auto ev1 = prepare_image(&data_pipeline, path_pair.second.c_str(),
                               input_img, sample_alloc_pool.input_data,
                               sample_alloc_pool.input_luma, true, false);
      data_pipeline.subtract_mean(sample_alloc_pool.input_luma, &ev1);
      sample_alloc_pool.w = (size_t)input_img.w;
      sample_alloc_pool.h = (size_t)input_img.h;
      context.block();
      context.raw_memory(sample_alloc_pool.input_data)->release();
      gpu_alloc.samples.push_back(sample_alloc_pool);
    }

    // train
    std::vector<float> last_weights1(data_pipeline.layer_1()->weight_size()),
        last_weights2(data_pipeline.layer_2()->weight_size()),
        last_weights3(data_pipeline.layer_3()->weight_size()),
        last_bias1(data_pipeline.layer_1()->bias_size()),
        last_bias2(data_pipeline.layer_2()->bias_size()),
        last_bias3(data_pipeline.layer_3()->bias_size());

    size_t samples_count = gpu_alloc.samples.size();
    std::vector<PerSampleAllocationPool> train_set(samples_count),
        validation_set(samples_count);
    bool train_error = false;
    for (size_t batch_id = 0; batch_id < batches_count; batch_id++) {
      // std::cout << "------ BATCH " << batch_id << " ------" << std::endl;
      context.block();
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
        if (std::isnan(train_squared_error)) {
          std::cout << "Error: squared error is NAN" << std::endl;
          train_error = true;
          break;
        }

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

      // if error happened we stop the training. Else copy parameters so if we
      // error in next batch we will still have proper values
      if (train_error) break;
      if (batch_id % 5 == 0) {
        std::cout << "(storing weights)" << std::endl;
        /* clang-format off */
      context.read_buffer(gpu_alloc.layer_1.weights, (void *)&last_weights1[0], true);
      context.read_buffer(gpu_alloc.layer_2.weights, (void *)&last_weights2[0], true);
      context.read_buffer(gpu_alloc.layer_3.weights, (void *)&last_weights3[0], true);
      context.read_buffer(gpu_alloc.layer_1.bias,    (void *)&last_bias1[0], true);
      context.read_buffer(gpu_alloc.layer_2.bias,    (void *)&last_bias2[0], true);
      context.read_buffer(gpu_alloc.layer_3.bias,    (void *)&last_bias3[0], true);
        /* clang-format on */
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

    context.block();
    if (train_error) {
      std::cout << "Training seemingly converged (very small error values "
                   "resulted in NAN)" << std::endl;
      data_pipeline.write_params_to_file(success_params_file,  //
                                         last_weights1, last_weights2,
                                         last_weights3,  //
                                         last_bias1, last_bias2, last_bias3);

      /* clang-format off */
      context.write_buffer(gpu_alloc.layer_1.weights, (void *)&last_weights1[0], true);
      context.write_buffer(gpu_alloc.layer_2.weights, (void *)&last_weights2[0], true);
      context.write_buffer(gpu_alloc.layer_3.weights, (void *)&last_weights3[0], true);
      context.write_buffer(gpu_alloc.layer_1.bias,    (void *)&last_bias1[0], true);
      context.write_buffer(gpu_alloc.layer_2.bias,    (void *)&last_bias2[0], true);
      context.write_buffer(gpu_alloc.layer_3.bias,    (void *)&last_bias3[0], true);
      /* clang-format on */
      context.block();
      execute_app(context, cfg, data_pipeline, gpu_alloc);
      context.block();

    } else {
      std::cout << "Training did not converge" << std::endl;
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
void get_training_samples(std::string dir_path,
                          std::vector<TrainSampleFiles>& target) {
  //
  std::vector<std::string> files;
  cnn_sr::utils::list_files(dir_path.c_str(), files);

  // split listed files by: base name, large/small
  std::unordered_map<std::string, TrainSampleFiles> files_by_base_name;
  for (auto file_path : files) {
    auto large_pos = file_path.rfind("_large.jpg");
    auto small_pos = file_path.rfind("_small.jpg");
    if (large_pos == std::string::npos && small_pos == std::string::npos) {
      if (file_path != "." && file_path != "..")
        std::cout << "'" << file_path << "' is not .jpg image. Skipping sample"
                  << std::endl;
    } else if (large_pos != std::string::npos) {
      auto& node = files_by_base_name[file_path.substr(0, large_pos)];
      node.first = dir_path + "\\" + file_path;
    } else {
      auto& node = files_by_base_name[file_path.substr(0, small_pos)];
      node.second = dir_path + "\\" + file_path;
    }
  }

  for (auto entry : files_by_base_name) {
    auto& pair = entry.second;
    if (pair.first.length() == 0 || pair.second.length() == 0) {
      std::cout << "Only 1 image for pair with name '" << entry.first
                << "'. Skipping sample" << std::endl;
    } else {
      // std::cout << entry.first << std::endl;
      target.push_back(pair);
    }
  }
}

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
