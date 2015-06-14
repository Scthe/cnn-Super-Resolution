#include <iostream>
#include <time.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // TODO remove ?

#include "opencl\Context.hpp"
#include "opencl\UtilsOpenCL.hpp"

#include "Config.hpp"

// http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html

#define OUT_CHANNELS 1

const char *cSourceFile = "src/kernel/greyscale.cl";
const char *img_path = "data/cold_reflection_by_lildream.jpg";
const char *img_path2 = "data/cold_2.png";

// http://www.thebigblob.com/gaussian-blur-using-opencl-and-the-built-in-images-textures/

void luma_extract(int argc, char **argv);

void cfg_tests();

int main(int argc, char **argv) {
  try {
    // luma_extract(argc, argv);
    cfg_tests();
  } catch (const std::exception &e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "DONE" << std::endl;
  exit(EXIT_SUCCESS);
}

void cfg_tests() {
  using namespace cnn_sr;
  ConfigReader reader;
  // reader.read("non_exists.json");
  // reader.read("tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___tooLong___.json");
  // reader.read("data\\config_err.json");
  Config cfg = reader.read("data\\config.json");

  std::cout << cfg << std::endl;
}

void luma_extract(int argc, char **argv) {
  using namespace opencl::utils;

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

  cl_image_format pixel_format;
  pixel_format.image_channel_order = CL_RGBA;
  pixel_format.image_channel_data_type = CL_UNSIGNED_INT8;
  auto gpu_image = context.create_image(CL_MEM_READ_WRITE,  //
                                        data.w, data.h, &pixel_format);
  context.write_image(gpu_image, data, true);

  // TODO change last arg to be pointer to data, now it is unused
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
