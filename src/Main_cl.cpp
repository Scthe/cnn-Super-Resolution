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

///
/// Util. structures
///
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

struct ParameterSet {
  // TODO move to ConfigBasedDataPipeline
  ParameterSet(size_t ws1, size_t ws2, size_t ws3,  //
               size_t bs1, size_t bs2, size_t bs3)
      : weights_1(ws1),
        weights_2(ws2),
        weights_3(ws3),
        bias_1(bs1),
        bias_2(bs2),
        bias_3(bs3) {}

  void store(opencl::Context& context, GpuAllocationPool& gpu_alloc) {
    context.read_buffer(gpu_alloc.layer_1.weights, (void*)&weights_1[0], true);
    context.read_buffer(gpu_alloc.layer_2.weights, (void*)&weights_2[0], true);
    context.read_buffer(gpu_alloc.layer_3.weights, (void*)&weights_3[0], true);
    context.read_buffer(gpu_alloc.layer_1.bias, (void*)&bias_1[0], true);
    context.read_buffer(gpu_alloc.layer_2.bias, (void*)&bias_2[0], true);
    context.read_buffer(gpu_alloc.layer_3.bias, (void*)&bias_3[0], true);
  }

  std::vector<float> weights_1, weights_2, weights_3;
  std::vector<float> bias_1, bias_2, bias_3;
};

///
/// Forward decl.
///
cl_event prepare_image(DataPipeline* const pipeline, const char* const,
                       ImageData&, opencl::MemoryHandle&, opencl::MemoryHandle&,
                       bool print = false);

void divide_samples(size_t validation_set_size, GpuAllocationPool&,
                    std::vector<PerSampleAllocationPool>& train_set,
                    std::vector<PerSampleAllocationPool>& validation_set);

typedef std::pair<std::string, std::string> TrainSampleFiles;

void get_training_samples(std::string, std::vector<TrainSampleFiles>&);

float execute_batch(bool backpropagate, ConfigBasedDataPipeline&,
                    GpuAllocationPool&, std::vector<PerSampleAllocationPool>);

void execute_forward(ConfigBasedDataPipeline&, GpuAllocationPool&,
                     const char* const in_path, const char* const out_path);

///
/// main
///
int main(int argc, char** argv) {
  std::srand(std::time(0));

  cnn_sr::utils::Argparse argparse("cnn", "????");
  /* clang-format off */
  argparse.add_argument("train").help("Train mode");
  argparse.add_argument("dry").help("Do not store result");
  argparse.add_argument("-c", "--config").required().help("CNN configuration");
  // argparse.add_argument("-p", "--parameters-file").help("Override parameters file provided in config");
  argparse.add_argument("-i", "--in").required().help("Image during forward, samples directory during training");
  argparse.add_argument("-o", "--out").help("Output file path (either result image or new parameters)");
  argparse.add_argument("-e", "--epochs").help("Number of epochs during training");
  /* clang-format on */

  if (!argparse.parse(argc, argv)) {
    exit(EXIT_SUCCESS);  // EXIT_FAILURE?
  }

  bool train = argparse.has_arg("train");
  bool dry = argparse.has_arg("dry");
  auto config_path = argparse.value("config");
  // auto pars_file_path = argparse.value("parameters-file");
  auto in_path = argparse.value("in");
  auto out_path = dry ? nullptr : argparse.value("out");
  size_t epochs;
  argparse.value("epochs", epochs);

  if (!dry && !out_path) {
    std::cout << "Either provide out path or do the dry run" << std::endl;
    exit(EXIT_FAILURE);
  }

  // print base info
  if (train) {
    std::cout << "Training mode, epochs: " << epochs << std::endl
              << "Training samples directory: " << in_path << std::endl
              << "Output: " << (out_path ? out_path : "-") << std::endl;
  } else {
    std::cout << "Forward mode" << std::endl
              << "Input image: " << in_path << std::endl
              << "Output: " << (out_path ? out_path : "-") << std::endl;
  }

  // other config variables
  const size_t validation_set_percent = 25;  // TODO move to cfg
  const size_t batches_between_params_store = 5;

  // read config
  ConfigReader reader;
  Config cfg = reader.read(config_path);
  std::cout << cfg << std::endl;

  // opencl context
  opencl::Context context;
  context.init();
  ConfigBasedDataPipeline data_pipeline(cfg, &context);
  data_pipeline.init();
  GpuAllocationPool gpu_alloc;

  if (!train) {
    execute_forward(data_pipeline, gpu_alloc, in_path, out_path);
    exit(EXIT_SUCCESS);
  }

  // training mode:
  // read training samples
  std::vector<TrainSampleFiles> train_sample_files;
  get_training_samples(in_path, train_sample_files);
  const size_t validation_set_size =
      (size_t)(train_sample_files.size() * validation_set_percent / 100.0f);
  if (validation_set_size == 0) {
    std::cout << "[WARNING] Validation set is empty" << std::endl;
  } else {
    std::cout << "validation_set_size: " << validation_set_size << "/"
              << train_sample_files.size() << " = "
              << (validation_set_size * 100.0f / train_sample_files.size())
              << "%" << std::endl;
  }

  // read & prepare images
  for (auto& path_pair : train_sample_files) {
    ImageData expected_output_img, input_img;
    PerSampleAllocationPool sample_alloc_pool;
    prepare_image(&data_pipeline, path_pair.first.c_str(), expected_output_img,
                  sample_alloc_pool.expected_output_data,
                  sample_alloc_pool.expected_output_luma);
    auto ev1 = prepare_image(&data_pipeline, path_pair.second.c_str(),
                             input_img, sample_alloc_pool.input_data,
                             sample_alloc_pool.input_luma);
    data_pipeline.subtract_mean(sample_alloc_pool.input_luma, nullptr, &ev1);
    sample_alloc_pool.w = (size_t)input_img.w;
    sample_alloc_pool.h = (size_t)input_img.h;
    context.block();
    context.raw_memory(sample_alloc_pool.input_data)->release();
    gpu_alloc.samples.push_back(sample_alloc_pool);
  }

  size_t samples_count = gpu_alloc.samples.size(),
         per_sample_px_count = gpu_alloc.samples[0].w * gpu_alloc.samples[0].h,
         validation_px_count = per_sample_px_count * validation_set_size,
         train_px_count =
             per_sample_px_count * (samples_count - validation_set_size);
  size_t l1_out_rows = gpu_alloc.samples[0].w - cfg.f1 + 1,
         l2_out_rows = l1_out_rows - cfg.f2 + 1,
         l3_out_rows = l2_out_rows - cfg.f3 + 1;
  ParameterSet last_good_parameter_set(data_pipeline.layer_1()->weight_size(),
                                       data_pipeline.layer_2()->weight_size(),
                                       data_pipeline.layer_3()->weight_size(),
                                       data_pipeline.layer_1()->bias_size(),
                                       data_pipeline.layer_2()->bias_size(),
                                       data_pipeline.layer_3()->bias_size());

  context.block();

  // train
  for (size_t batch_id = 0; batch_id < epochs; batch_id++) {
    std::vector<PerSampleAllocationPool> train_set(samples_count);
    std::vector<PerSampleAllocationPool> validation_set(samples_count);
    divide_samples(validation_set_size, gpu_alloc, train_set, validation_set);

    float train_squared_error =
        execute_batch(true, data_pipeline, gpu_alloc, train_set);

    // if error happened we stop the training.
    if (std::isnan(train_squared_error)) {
      std::cout << "Error: squared error is NAN" << std::endl;
      break;
    }
    // Copy parameters so if we error in next batch we will still have proper
    // values
    if (batch_id > 1 && batch_id % batches_between_params_store == 0) {
      std::cout << "(storing weights)" << std::endl;
      last_good_parameter_set.store(context, gpu_alloc);
    }

    data_pipeline.update_parameters(gpu_alloc.layer_1, gpu_alloc.layer_2,
                                    gpu_alloc.layer_3, train_set.size());
    context.block();

    /* clang-format off */
    // ConfigBasedDataPipeline& d = data_pipeline;
    // d.print_buffer(gpu_alloc.layer_1.bias, "layer 1 bias", 1);
    // d.print_buffer(gpu_alloc.layer_2.bias, "layer 2 bias", 1);
    // d.print_buffer(gpu_alloc.layer_3.bias, "layer 3 bias", 1);
    // d.print_buffer(gpu_alloc.layer_1.accumulating_grad_b, "layer 1 bias gradients", 1);
    // d.print_buffer(gpu_alloc.layer_2.accumulating_grad_b, "layer 2 bias gradients", 1);
    // d.print_buffer(gpu_alloc.layer_3.accumulating_grad_b, "layer 3 bias gradients", 1);

    // d.print_buffer(gpu_alloc.layer_1.weights, "layer 1 weights", cfg.f1*cfg.f1);
    // d.print_buffer(gpu_alloc.layer_2.weights, "layer 2 weights", cfg.f2*cfg.f2);
    // d.print_buffer(gpu_alloc.layer_3.weights, "layer 3 weights", cfg.f3*cfg.f3);
    // d.print_buffer(gpu_alloc.layer_1.accumulating_grad_w, "layer 1 weight gradients", cfg.f1*cfg.f1);
    // d.print_buffer(gpu_alloc.layer_2.accumulating_grad_w, "layer 2 weight gradients", cfg.f2*cfg.f2);
    // d.print_buffer(gpu_alloc.layer_3.accumulating_grad_w, "layer 3 weight gradients", cfg.f3*cfg.f3);

    // d.print_buffer(gpu_alloc.layer_1.output, "layer 1 out", l1_out_rows);
    // d.print_buffer(gpu_alloc.layer_2.output, "layer 2 out", l2_out_rows);
    // d.print_buffer(gpu_alloc.layer_3.output, "layer 3 out", l3_out_rows);
    // d.print_buffer(gpu_alloc.layer_1.deltas, "layer 1 deltas", l1_out_rows);
    // d.print_buffer(gpu_alloc.layer_2.deltas, "layer 2 deltas", l2_out_rows);
    // d.print_buffer(gpu_alloc.layer_3.deltas, "layer 3 deltas", l3_out_rows);
    /* clang-format on */

    float validation_squared_error =
        execute_batch(false, data_pipeline, gpu_alloc, validation_set);

    // (we are printing per pixel values because they are easier to remember)
    float mean_train_err = train_squared_error / train_set.size(),
          mean_valid_err = validation_squared_error / validation_set.size();
    // std::cout << "[" << batch_id << "] "  //
    // << "mean train error: " << mean_train_err << " ("
    // << (mean_train_err / train_px_count) << " per px)" << std::endl;

    std::cout << "[" << batch_id << "] "  //
              /*<< "mean train error: " << mean_train_err << " ("
              << (mean_train_err / train_px_count) << " per px), "*/
              << "mean validation error: " << mean_valid_err << " ("
              << (mean_valid_err / validation_px_count) << " per px)"
              << std::endl;

    context.block();
  }

  if (out_path) {
    data_pipeline.write_params_to_file(out_path,
                                       last_good_parameter_set.weights_1,  //
                                       last_good_parameter_set.weights_2,
                                       last_good_parameter_set.weights_3,
                                       last_good_parameter_set.bias_1,
                                       last_good_parameter_set.bias_2,  //
                                       last_good_parameter_set.bias_3);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}

///
/// Forward
///
void execute_forward(ConfigBasedDataPipeline& data_pipeline,
                     GpuAllocationPool& gpu_alloc, const char* const in_path,
                     const char* const out_path) {
  auto context = data_pipeline.context();
  auto cfg = data_pipeline.config();

  // read input image
  ImageData input_img;
  opencl::MemoryHandle input_data = gpu_nullptr;
  opencl::MemoryHandle input_luma = gpu_nullptr;
  auto ev1 =
      prepare_image(&data_pipeline, in_path, input_img, input_data, input_luma);
  float mean;
  data_pipeline.subtract_mean(input_luma, &mean, &ev1);
  size_t w = input_img.w, h = input_img.h,  //
      luma_w = w - cfg->total_padding(),    //
      luma_h = h - cfg->total_padding();
  context->block();

  // process with layers
  auto forward_ev = data_pipeline.forward(gpu_alloc.layer_1,  //
                                          gpu_alloc.layer_2,  //
                                          gpu_alloc.layer_3,  //
                                          input_luma, w, h);
  // dbg output read
  // data_pipeline.print_buffer(gpu_alloc.layer_1.output, "layer 1", h);
  // data_pipeline.print_buffer(gpu_alloc.layer_2.output, "layer 2", h);
  // data_pipeline.print_buffer(gpu_alloc.layer_3.output, "OUT", h);

  if (out_path) {
    data_pipeline.write_result_image(out_path, input_img, input_data,
                                     input_luma, mean,  //
                                     gpu_alloc.layer_3.output, luma_w, luma_h);
  }
}

///
/// Training
///
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

float execute_batch(bool backpropagate, ConfigBasedDataPipeline& data_pipeline,
                    GpuAllocationPool& gpu_alloc,
                    std::vector<PerSampleAllocationPool> sample_set) {
  auto context = data_pipeline.context();
  auto cfg = data_pipeline.config();

  float squared_error = 0;
  for (PerSampleAllocationPool& sample : sample_set) {
    const size_t w = sample.w, h = sample.h;
    // process with layers
    auto forward_ev = data_pipeline.forward(gpu_alloc.layer_1,  //
                                            gpu_alloc.layer_2,  //
                                            gpu_alloc.layer_3,  //
                                            sample.input_luma, w, h);
    // squared difference
    squared_error += data_pipeline.squared_error(
        sample.input_luma, gpu_alloc.layer_3.output, w, h, &forward_ev);
    if (std::isnan(squared_error)) {
      return squared_error;
    }

    if (backpropagate) {
      auto weight_decay_value = data_pipeline.weight_decay(
          gpu_alloc.layer_1, gpu_alloc.layer_2, gpu_alloc.layer_3,
          cfg->weight_decay_parameter, &forward_ev);
      data_pipeline.backpropagate(gpu_alloc.layer_1,  //
                                  gpu_alloc.layer_2,  //
                                  gpu_alloc.layer_3,  //
                                  sample.input_luma,
                                  sample.expected_output_luma, w, h,
                                  weight_decay_value);
    }
    context->block();
  }
  return squared_error;
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
                       opencl::MemoryHandle& gpu_luma_handle, bool print) {
  bool normalize_luma = true;
  if (print) std::cout << "loading image '" << file_path << "'";
  opencl::utils::load_image(file_path, img_data);  // TODO should throw
  if (print) {
    std::cout << ", size: " << img_data.w << "x" << img_data.h << "x"
              << img_data.bpp << std::endl;
  }

  // extract luma channel
  return pipeline->extract_luma(img_data, gpu_data_handle, gpu_luma_handle,
                                normalize_luma);
}
