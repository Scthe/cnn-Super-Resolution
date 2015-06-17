#ifndef LAYER_EXECUTOR_H
#define LAYER_EXECUTOR_H

#include <cstddef>  // for size_t
#include <vector>

typedef struct _cl_event* cl_event;

namespace opencl {
class Kernel;
struct MemoryHandler;
}

namespace cnn_sr {

struct Config;
struct LayerData;

/**
 * Class responsible for taking LayerData and feeding it ti the gpu kernel
 */
class LayerExecutor {
 public:
  cl_event operator()(opencl::Kernel& kernel, const LayerData& data,
                      std::vector<float>& input,
                      opencl::MemoryHandler*& gpu_buf_out, size_t input_w,
                      size_t input_h, cl_event* ev = nullptr);

 private:
  void pre_exec_validation(const LayerData&, std::vector<float>&, size_t,
                           size_t);
};
}

#endif /* LAYER_EXECUTOR_H   */
