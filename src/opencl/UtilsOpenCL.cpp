#include <iostream>
#include <stdio.h>
#include <strings.h>
#include <stdexcept>
#include <algorithm>  // f.e. std::minmax_element

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "UtilsOpenCL.hpp"
#include "Kernel.hpp"
#include "Context.hpp"

size_t closest_power_of_2(int x);

namespace opencl {
namespace utils {

char const *device_type_str[9] = {
    "-",
    "default",  // 1
    "CPU",      // 2
    "-",
    "GPU",  // 4
    "-",           "-", "-",
    "Accelerator",  // 8
};

char *load_file(const char *cFilename, const char *cPreamble,
                size_t *szFinalLength) {
  FILE *pFileStream = NULL;
  size_t szSourceLength;

#ifdef _MSC_VER  // Visual studio
  if (fopen_s(&pFileStream, cFilename, "rb") != 0) {
    return NULL;
  }
#else  // Linux version
  pFileStream = fopen(cFilename, "rb");
  if (pFileStream == 0) {
    return NULL;
  }
#endif

  size_t szPreambleLength = strlen(cPreamble);

  // get the length of the source code
  fseek(pFileStream, 0, SEEK_END);
  szSourceLength = ftell(pFileStream);
  fseek(pFileStream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in
  char *cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
  memcpy(cSourceString, cPreamble, szPreambleLength);
  if (fread((cSourceString) + szPreambleLength, szSourceLength, 1,
            pFileStream) != 1) {
    fclose(pFileStream);
    free(cSourceString);
    return 0;
  }

  // close the file and return the total length of the combined
  // (preamble + source) string
  fclose(pFileStream);
  if (szFinalLength != 0) {
    *szFinalLength = szSourceLength + szPreambleLength;
  }
  cSourceString[szSourceLength + szPreambleLength] = '\0';

  return cSourceString;
}

///
/// images
/// TODO move from OpenCL to general utils
///

ImageData::ImageData() : w(0), h(0), bpp(0), data(nullptr) {}

ImageData::ImageData(int w, int h, int bpp, unsigned char *data)
    : w(w), h(h), bpp(bpp), data(data), read_from_file(false) {}

ImageData::~ImageData() {
  if (data && read_from_file) stbi_image_free(data);
}

void load_image(const char *filename, ImageData &data) {
  data.data = stbi_load(filename, &data.w, &data.h, &data.bpp, 4);
  // TODO CHECK_ALLOCATION(data.data);
}

int write_image(const char *filename, ImageData &data) {
  return stbi_write_png(filename, data.w, data.h, data.bpp, data.data, 0);
}

void write_image(const char *const file_path, float *source,  //
                 size_t w, size_t h) {
  size_t px_cnt = w * h;
  // normalize values: 0..1
  auto min_max_it = std::minmax_element(source, source + px_cnt);
  float min = *min_max_it.first, max = *min_max_it.second,
        norm_factor = max - min;
  for (size_t i = 0; i < px_cnt; i++) {
    source[i] = (source[i] - min) / norm_factor;
  }

  std::cout << "writing image(" << w << "x" << h << ") to: '" << file_path
            << "'" << std::endl;
  std::vector<unsigned char> data(px_cnt * 3);
  for (size_t row = 0; row < h; row++) {
    for (size_t col = 0; col < w; col++) {
      size_t idx = row * w + col;
      float val = source[idx] * 255;
      for (size_t k = 0; k < 3; k++) {
        data[idx * 3 + k] = (unsigned char)val;
      }
    }
  }

  ImageData dd(w, h, sizeof(unsigned char) * 3, &data[0]);
  opencl::utils::write_image(file_path, dd);
}

///
/// misc
///

void work_sizes(const opencl::Kernel &kernel, size_t dim,
                size_t *global_work_size, size_t *local_work_size, size_t *work,
                bool print) {
  if (dim == 0 || dim > 3) {
    throw std::runtime_error("Work dimesions should be 1,2 or 3");
  }

  auto context = kernel.get_context();
  auto device = context->device();
  auto max_local =
      std::min(device.max_work_group_size, kernel.get_max_work_group_size());
  auto max_device_local_size = device.work_items_for_dims;

  // global_work_size
  for (size_t i = 0; i < dim; i++) {
    global_work_size[i] = closest_power_of_2(static_cast<int>(work[i]));
  }

  // local_work_size
  // we are doing round robin (see to_update variable) multiplying each
  // dimension by 2 each time. It may not work that good for:
  // max_device_local_size = [1024, 1024, 1], since it stops after 3 iterations
  // On the other note I've had to look up syntax to do{..}while(...);
  size_t tmp[3] = {1, 1, 1}, local_dims_multiplied = 1, to_update = 0;
  bool satisfies_conditions;
  do {
    // copy last correct configuration to local
    memcpy(local_work_size, tmp, dim * sizeof(float));
    tmp[to_update] *= 2;
    local_dims_multiplied *= 2;
    satisfies_conditions = tmp[to_update] <= max_device_local_size[to_update] &&
                           tmp[to_update] <= global_work_size[to_update] &&
                           local_dims_multiplied <= max_local;
    to_update = (to_update + 1) % dim;
  } while (satisfies_conditions);

  bool ok = true;
  for (size_t i = 0; i < dim; i++) {
    ok &= global_work_size[i] >= local_work_size[i];
    ok &= local_work_size[i] > 0;
  }

  if (!ok) {
    char buf[255];
    snprintf(buf, 255,
             "Tried to create nonstandard work dimensions: global=[%d,%d,%d], "
             "local=[%d,%d,%d]",
             global_work_size[0], (dim > 1 ? global_work_size[1] : 1),
             (dim == 3 ? global_work_size[2] : 1),  //
             local_work_size[0], (dim > 1 ? local_work_size[1] : 1),
             (dim == 3 ? local_work_size[2] : 1));
    throw std::runtime_error(buf);
  }

  if (print) {
    std::cout << "global work size: ["                        //
              << global_work_size[0] << ", "                  //
              << (dim > 1 ? global_work_size[1] : 1) << ", "  //
              << (dim == 3 ? global_work_size[2] : 1) << "]" << std::endl;
    std::cout << "local work size: ["                        //
              << local_work_size[0] << ", "                  //
              << (dim > 1 ? local_work_size[1] : 1) << ", "  //
              << (dim == 3 ? local_work_size[2] : 1) << "]" << std::endl;
  }
}

const char *get_opencl_error_str(cl_int errorCode) {
#define DECLARE_ERROR(err) \
  case (err):              \
    return #err

  switch (errorCode) {
    DECLARE_ERROR(CL_SUCCESS);
    DECLARE_ERROR(CL_DEVICE_NOT_FOUND);
    DECLARE_ERROR(CL_DEVICE_NOT_AVAILABLE);
    DECLARE_ERROR(CL_COMPILER_NOT_AVAILABLE);
    DECLARE_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES - either running out of memory or possible "
             "watchdog exception. See f.e "
             "https://devtalk.nvidia.com/default/topic/471020/"
             "driver-crashs-while-opencl-app-is-running/";
      DECLARE_ERROR(CL_OUT_OF_HOST_MEMORY);
      DECLARE_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
      DECLARE_ERROR(CL_MEM_COPY_OVERLAP);
      DECLARE_ERROR(CL_IMAGE_FORMAT_MISMATCH);
      DECLARE_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
      DECLARE_ERROR(CL_BUILD_PROGRAM_FAILURE);
      DECLARE_ERROR(CL_MAP_FAILURE);
      DECLARE_ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
      DECLARE_ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      DECLARE_ERROR(CL_INVALID_VALUE);
      DECLARE_ERROR(CL_INVALID_DEVICE_TYPE);
      DECLARE_ERROR(CL_INVALID_PLATFORM);
      DECLARE_ERROR(CL_INVALID_DEVICE);
      DECLARE_ERROR(CL_INVALID_CONTEXT);
      DECLARE_ERROR(CL_INVALID_QUEUE_PROPERTIES);
      DECLARE_ERROR(CL_INVALID_COMMAND_QUEUE);
      DECLARE_ERROR(CL_INVALID_HOST_PTR);
      DECLARE_ERROR(CL_INVALID_MEM_OBJECT);
      DECLARE_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
      DECLARE_ERROR(CL_INVALID_IMAGE_SIZE);
      DECLARE_ERROR(CL_INVALID_SAMPLER);
      DECLARE_ERROR(CL_INVALID_BINARY);
      DECLARE_ERROR(CL_INVALID_BUILD_OPTIONS);
      DECLARE_ERROR(CL_INVALID_PROGRAM);
      DECLARE_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
      DECLARE_ERROR(CL_INVALID_KERNEL_NAME);
      DECLARE_ERROR(CL_INVALID_KERNEL_DEFINITION);
      DECLARE_ERROR(CL_INVALID_KERNEL);
      DECLARE_ERROR(CL_INVALID_ARG_INDEX);
      DECLARE_ERROR(CL_INVALID_ARG_VALUE);
      DECLARE_ERROR(CL_INVALID_ARG_SIZE);
      DECLARE_ERROR(CL_INVALID_KERNEL_ARGS);
      DECLARE_ERROR(CL_INVALID_WORK_DIMENSION);
      DECLARE_ERROR(CL_INVALID_WORK_GROUP_SIZE);
      DECLARE_ERROR(CL_INVALID_WORK_ITEM_SIZE);
      DECLARE_ERROR(CL_INVALID_GLOBAL_OFFSET);
      DECLARE_ERROR(CL_INVALID_EVENT_WAIT_LIST);
      DECLARE_ERROR(CL_INVALID_EVENT);
      DECLARE_ERROR(CL_INVALID_OPERATION);
      DECLARE_ERROR(CL_INVALID_GL_OBJECT);
      DECLARE_ERROR(CL_INVALID_BUFFER_SIZE);
      DECLARE_ERROR(CL_INVALID_MIP_LEVEL);
      DECLARE_ERROR(CL_INVALID_GLOBAL_WORK_SIZE);
      // DECLARE_ERROR(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
      // DECLARE_ERROR(CL_PLATFORM_NOT_FOUND_KHR);
      // DECLARE_ERROR(CL_INVALID_PROPERTY_EXT);
      // DECLARE_ERROR(CL_DEVICE_PARTITION_FAILED_EXT);
      // DECLARE_ERROR(CL_INVALID_PARTITION_COUNT_EXT);
      DECLARE_ERROR(CL_INVALID_PROPERTY);
    default:
      return "unknown error code";
  }
#undef DECLARE_ERROR
}

//
}
}

size_t closest_power_of_2(int x) {
  if (x < 0) return 0;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return static_cast<size_t>(x + 1);
}
