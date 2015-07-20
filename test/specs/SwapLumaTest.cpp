#include "TestSpecsDeclarations.hpp"

#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"

/*
 * Just run the kernel and see if the output is ~what You would expect.
 * This is quite easy if luma that You swap into has distinctive pattern.
 * You could also try generating expected output through python
 * (see LumaTests_script.py), but it uses some weird sampling method,
 * which means that result image is slighly blurred.
 */

namespace test {
namespace specs {

///
/// PIMPL
///
struct SwapLumaTestImpl {
  const size_t padding = 10;
  // const char * const test_image = "test/data/color_grid.png";
  const char *const input_img = "test/data/color_grid2.jpg";
  const char *const expected_img = "test/data/color_grid2_luma_swapped.png";
};

///
/// SwapLumaTest
///

TEST_SPEC_PIMPL(SwapLumaTest)

void SwapLumaTest::init() {}

size_t SwapLumaTest::data_set_count() { return 1; }

std::string SwapLumaTest::name(size_t) { return "Swap luma test"; }

bool SwapLumaTest::operator()(size_t, cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  auto context = pipeline->context();

  opencl::utils::ImageData img;
  load_image(_impl->input_img, img);

  // generate luma to swap into
  size_t luma_w = img.w - 2 * _impl->padding,
         luma_h = img.h - 2 * _impl->padding, new_luma_size = luma_w * luma_w;
  std::vector<float> new_luma(new_luma_size);
  for (size_t i = 0; i < new_luma_size; i++) {
    new_luma[i] = i * 1.0f / new_luma_size;
  }

  opencl::MemoryHandle gpu_buf_raw_img = gpu_nullptr,
                       gpu_buf_target = gpu_nullptr;
  auto gpu_buf_luma =
      context->allocate(CL_MEM_READ_ONLY, sizeof(cl_float) * new_luma.size());
  context->write_buffer(gpu_buf_luma, (void *)&new_luma[0], true);

  // run
  pipeline->swap_luma(img, gpu_buf_raw_img, gpu_buf_luma, gpu_buf_target,
                      luma_w, luma_h);
  // check
  opencl::utils::ImageData expected_img;
  load_image(_impl->expected_img, expected_img);
  size_t result_w = expected_img.w, result_h = expected_img.h,
         result_size = result_w * result_h * 3;  // 3 channels
  std::vector<unsigned char> result(result_size);
  context->read_buffer(gpu_buf_target, (void *)&result[0], true);

  // dump image - only for debug !!!
  // opencl::utils::ImageData res_img(result_w, result_h, 3, &result[0]);
  // opencl::utils::write_image("dbg.png", res_img);

  for (size_t y = 0; y < result_h; y++) {
    for (size_t x = 0; x < result_w; x++) {
      for (size_t ch = 0; ch < 3; ch++) {
        // NOTE: expected_img has 4 channels !
        size_t idx1 = y * result_w + x;
        int r = static_cast<int>(result[idx1 * 3 + ch]);
        int e = static_cast<int>(expected_img.data[idx1 * 4 + ch]);
        // std::cout << "[" << idx1 << "] expected >\t" << e << "\tgot> " << r
        // << std::endl;
        assert_equals(e, r);
      }
    }
  }

  return true;
}

//
//
}  // namespace specs
}  // namespace test
