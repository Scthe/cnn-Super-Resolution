#ifndef UTILS_OPENCL_H
#define UTILS_OPENCL_H

#include "CL/opencl.h"

namespace opencl {

class Kernel;

namespace utils {

/**
 * From stb_image documentation:
 *
 * The return value from an image loader is an 'unsigned char *' which points
 * to the pixel data, or NULL on an allocation failure or if the image is
 * corrupt or invalid. The pixel data consists of *y scanlines of *x pixels,
 * with each pixel consisting of N interleaved 8-bit components; the first
 * pixel pointed to is top-left-most in the image. There is no padding between
 * image scanlines or between pixels, regardless of format. The number of
 * components N is 'req_comp' if req_comp is non-zero, or *comp otherwise.
 * If req_comp is non-zero, *comp has the number of components that _would_
 * have been output otherwise. E.g. if you set req_comp to 4, you will always
 * get RGBA output, but you can check *comp to see if it's trivially opaque
 * because e.g. there were only 3 channels in the source image.
 */
struct ImageData {
  ImageData();
  ImageData(int,int,int,unsigned char*);
  ~ImageData();
  // TODO do not allow copy !!!

  int w, h;
  int bpp;  // bytes per pixel
  unsigned char* data;
private:
  bool read_from_file = true;
};

/**
 * cl_device_type is a number so we will change it to string
 */
extern char const* device_type_str[9];

/**
 * Loads a Program file and prepends the cPreamble to the code.
 * @param  cFilename     program filename
 * @param  cPreamble     code that is prepended to the loaded file, typically \
a set of #defines or a header
 * @param  szFinalLength returned length of the code string
 * @return               the source string if succeeded, 0 otherwise
 */
char* load_file(const char* cFilename, const char* cPreamble,
                size_t* szFinalLength);

void load_image(const char*, ImageData&);

int write_image(const char*, ImageData&);

/**
 * Due too different possible resolutions we may have to recalculate this each
 * time.
 *
 * NOTE: this solution tries to maximize work items per group, as most of
 *kernels have some __local related optimizations
 *
 * @param kernel           kernel to execute
 * @param dims             work dimensions: 1 for linear, 2 for 2D, 3 for 3D
 * @param global_work_size to be filled size: dims
 * @param local_work_size  to be filled size: dims
 * @param work             real work size f.e. array length, image dimesions
 *                         etc. size: dims
 */
void work_sizes(const opencl::Kernel&, size_t dims, size_t* global_work_size,
                size_t* local_work_size, size_t* work, bool print = false);

/**
 * convert error code to string
 *
 * @param  cl_int :cl_int, error code
 * @return        :string
 */
const char* get_opencl_error_str(cl_int);
}
}

#endif /* UTILS_OPENCL_H   */
