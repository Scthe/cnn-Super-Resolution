#include "TestSpecsDeclarations.hpp"

#include "../../src/opencl/UtilsOpenCL.hpp"
#include "../../src/DataPipeline.hpp"

auto test_image = "test/data/color_grid.png";

namespace test {
namespace specs {

///
/// Data set
///
struct ExtractLumaDataSet : DataSet {
  ExtractLumaDataSet(bool n, std::string name) : DataSet(name), normalize(n) {}
  bool normalize;
};

ExtractLumaDataSet data_sets[2] = {ExtractLumaDataSet(true, "normalized"),
                                   ExtractLumaDataSet(false, "not normalized")};
size_t ExtractLumaTest::data_set_count() { return 2; }

///
/// PIMPL
///
struct ExtractLumaTestImpl {
  const size_t data_size[2] = {5, 5};
  const float output[25] = {0.000f, 1.000f, 0.812f, 0.853f, 0.437f,  //
                            0.170f, 0.701f, 0.413f, 0.886f, 0.787f,  //
                            0.430f, 0.299f, 0.587f, 0.114f, 0.707f,  //
                            0.670f, 0.745f, 0.853f, 0.745f, 0.299f,
                            0.810f, 0.588f, 0.859f, 0.593f, 0.702f};
};

///
/// ExtractLumaTest
///

TEST_SPEC_PIMPL(ExtractLumaTest)

void ExtractLumaTest::init() {}

std::string ExtractLumaTest::name(size_t data_set_id) {
  assert_data_set_ok(data_set_id);
  return "Extract luma test - " + data_sets[data_set_id].name;
}

bool ExtractLumaTest::operator()(size_t data_set_id,
                                 cnn_sr::DataPipeline *const pipeline) {
  assert_not_null(pipeline);
  assert_data_set_ok(data_set_id);
  bool normalize = data_sets[data_set_id].normalize;
  auto _context = pipeline->context();

  opencl::utils::ImageData data;
  load_image(test_image, data);
  this->assert_true(
      _impl->data_size[0] * _impl->data_size[1] == (size_t)(data.w * data.h),
      "Vector of 1st layer's input values should be at least as big as test"
      " image");

  opencl::MemoryHandle gpu_buf_raw_img = gpu_nullptr,
                       gpu_buf_luma = gpu_nullptr;
  auto finish_token =
      pipeline->extract_luma(data, gpu_buf_raw_img, gpu_buf_luma, normalize);

  float cpu_buf[25];
  _context->read_buffer(gpu_buf_luma, (void *)cpu_buf, true, &finish_token, 1);

  for (int i = 0; i < data.w * data.h; i++) {
    float expected = _impl->output[i];
    if (!normalize) expected *= 255;
    // std::cout << i << ": " << cpu_buf[i] << "\t" << expected << std::endl;
    assert_equals(expected, cpu_buf[i]);
  }
  return true;
}

//
//
}  // namespace specs
}  // namespace test
