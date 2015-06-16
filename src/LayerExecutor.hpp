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

  /**
   * Due too different possible resolutions we may have to recalculate this each
   * time.
   * Implementation note: we assume that device's address_bits can hold the
   * number of range w*h. For example if address_bits==32 then we would need
   * image bigger then 2^16 in width and height for this condition to fail.
   * There is appropriate check in Kernel class.
   *
   * @param global_work_size float array of size 2
   * @param local_work_size float array of size 2
   */
  void work_sizes(const opencl::Kernel&, size_t* global_work_size,
                  size_t* local_work_size, size_t w, size_t h);
};
}

#endif /* LAYER_EXECUTOR_H   */
