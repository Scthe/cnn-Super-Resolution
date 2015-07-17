#include "TestSpecsDeclarations.hpp"

#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>

#include "json/gason.h"

#include "../../src/DataPipeline.hpp"
#include "../../src/LayerData.hpp"
#include "../../src/Utils.hpp"

auto test_data_file = "test/data/test_cases.json";

/* clang-format off */
/*
 *
 * NOTE: use LayerTest_script.R to generate expected output values
 *
 *
 *  Test data schema description (values for each layer provided after '/'):
 *
 *  n_prev_filter_cnt    := INT, filter count for previous layer, values: 1/n1/n2
 *  current_filter_count := INT, filter count for this layer, values: n1/n2/1
 *  f_spatial_size       := INT, spatial size, values: f1/f2/f3
 *  input_w              := INT, input dimensions
 *  input_h              := INT, input dimensions
 *  input                := VECTOR[FLOAT], min size: input_w * input_h * n_prev_filter_cnt.
 *                           Each column for different filter(from 1 to n_prev_filter_cnt).
 *                           Each row for different point in range 0..input_w*input_h
 *  output               := VECTOR[FLOAT], min size: out_w * out_h * current_filter_count
 *                           Expected output
 *  weights              := VECTOR[FLOAT], min size: f_spatial_size^2 * n_prev_filter_cnt * current_filter_count
 *                           There are f_spatial_size paragraphs
 *                           Each paragraph consists of f_spatial_size lines, representing 1 row.
 *                           Each row contains current_filter_count*n_prev_filter_cnt numbers,
 *                           grouped by n_prev_filter_cnt (n_prev_filter_cnt groups,
 *                           current_filter_count numbers per each group).
 *  bias                 := VECTOR[FLOAT], min size: current_filter_count
 *
 *
 * calcutated values:
 *   out_w := input_w - f_spatial_size + 1
 *   out_h := input_h - f_spatial_size + 1
 */
/* clang-format on */

namespace test {
namespace specs {

///
/// Data set
///
struct LayerDataSet : DataSet {
  size_t n_prev_filter_cnt,  //
      current_filter_count,  //
      f_spatial_size,        //
      input_w, input_h;
  std::vector<float> input;
  std::vector<float> output;
  std::vector<float> weights;
  std::vector<float> bias;
};

///
/// PIMPL
///
struct LayerTestImpl {
  bool read_test_data_from_file(char const* const file);

  std::vector<LayerDataSet> data_sets;
};

///
/// LayerTest
///

TEST_SPEC_PIMPL(LayerTest)

void LayerTest::init() {
  auto status = _impl->read_test_data_from_file(test_data_file);
  if (!status) {
    exit(EXIT_FAILURE);
  }
}

size_t LayerTest::data_set_count() { return _impl->data_sets.size(); }

std::string LayerTest::name(size_t data_set_id) {
  if (data_set_count() == 0) {
    return "Layer test - no data sets provided";
  }
  assert_data_set_ok(data_set_id);
  return "Layer test - " + _impl->data_sets[data_set_id].name;
}

bool LayerTest::operator()(size_t data_set_id,
                           cnn_sr::DataPipeline* const pipeline) {
  if (data_set_count() == 0) return false;

  assert_not_null(pipeline);
  assert_data_set_ok(data_set_id);
  auto data = &_impl->data_sets[data_set_id];
  auto _context = pipeline->context();

  // convert layer test definition to cnn_sr::LayerData object
  cnn_sr::LayerData layer_data(data->n_prev_filter_cnt,
                               data->current_filter_count,
                               data->f_spatial_size);
  layer_data.set_weights(&data->weights[0]);
  layer_data.set_bias(&data->bias[0]);

  size_t out_dim[2];
  layer_data.get_output_dimensions(out_dim, data->input_w, data->input_h);

  // alloc input
  cnn_sr::CnnLayerGpuAllocationPool gpu_alloc;
  auto gpu_buf_in = _context->allocate(CL_MEM_WRITE_ONLY,
                                       sizeof(cl_float) * data->input.size());
  _context->write_buffer(gpu_buf_in, (void*)&data->input[0], true);

  // create kernel & run
  auto kernel =
      pipeline->create_layer_kernel(layer_data);
  pipeline->execute_layer(*kernel, layer_data, gpu_alloc, gpu_buf_in,
                          data->input_w, data->input_h);
  assert_equals(pipeline, data->output, gpu_alloc.output);

  return true;
}

//
//
//

bool read_layer_data(const JsonValue& object, LayerDataSet& data) {
  // ASSERT(object.getTag() == JSON_TAG_OBJECT);

  for (auto node : object) {
    JSON_READ_UINT(node, data, n_prev_filter_cnt)
    JSON_READ_UINT(node, data, f_spatial_size)
    JSON_READ_UINT(node, data, current_filter_count)
    JSON_READ_UINT(node, data, input_w)
    JSON_READ_UINT(node, data, input_h)
    JSON_READ_NUM_ARRAY(node, data, input)
    JSON_READ_NUM_ARRAY(node, data, output)
    JSON_READ_NUM_ARRAY(node, data, weights)
    JSON_READ_NUM_ARRAY(node, data, bias)
  }

  return true;
}

bool LayerTestImpl::read_test_data_from_file(char const* const file) {
  std::cout << "Loading layer test data from: '" << file << "'" << std::endl;

  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  cnn_sr::utils::read_json_file(file, value, allocator, source, JSON_OBJECT);

  bool read_status = true;
  if (value.getTag() == JSON_OBJECT) {
    for (auto object : value) {
      if (object->value.getTag() != JSON_OBJECT) continue;
      // std::cout << object->key << std::endl;
      data_sets.push_back(LayerDataSet());
      LayerDataSet* ptr = &data_sets[data_sets.size() - 1];
      ptr->name = object->key;
      read_status &= read_layer_data(object->value, *ptr);
    }
  }

  return read_status;
}

//
//
}  // namespace specs
}  // namespace test
