#include <iostream>
#include <stdio.h>
#include <strings.h>


#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "UtilsOpenCL.hpp"

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
///

ImageData::ImageData() : w(0), h(0), bpp(0), data(nullptr) {}

ImageData::~ImageData() {
  if (data) stbi_image_free(data);
}

void load_image(const char *filename, ImageData &data) {
  data.data = stbi_load(filename, &data.w, &data.h, &data.bpp, 4);
  // TODO CHECK_ALLOCATION(data.data);
}

int write_image(const char *filename, ImageData &data) {
  return stbi_write_png(filename, data.w, data.h, data.bpp, data.data, 0);
}

///
/// misc
///

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
      return "CL_OUT_OF_RESOURCES - possible watchdog exception"
             "see f.e "
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
