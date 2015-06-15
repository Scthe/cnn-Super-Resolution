#include "TestDataProvider.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>

#include "json/gason.h"

#include "../src/Utils.hpp"

/* clang-format off */
/*
 *
 *  Test data schema description (values for each layer provided after '/'):
 *
 *  n_prev_filter_cnt    := INT, filter count for previous layer, values: 1/n1/n2
 *  current_filter_count := INT, filter count for this layer, values: n1/n2/1
 *  f_spatial_size       := INT, spatial size, values: f1/f2/f3
 *  input_w              := INT, input dimensions
 *  input_h              := INT, input dimensions
 *  result_multiply      := INT, optional - multiply result by const instead of using ReLU
 *  input                := VECTOR[FLOAT], min size: input_w * input_h * n_prev_filter_cnt.
 *  												 Each column for different filter(from 1 to n_prev_filter_cnt).
 *  												 Each row for different point in range 0..input_w*input_h
 *  output               := VECTOR[FLOAT], min size: out_w * out_h * current_filter_count
 *                           Expected output
 *  weights              := VECTOR[FLOAT], min size: f_spatial_size^2 * n_prev_filter_cnt
 *  												 Each column for different filter(from 1 to n_prev_filter_cnt)
 *  												 Each row for different point in range 0..f_spatial_size^2
 *  												 Each paragraph is 1 row of points  (f_spatial_size points)
 *  bias                 := VECTOR[FLOAT], min size: current_filter_count
 *
 *
 * calcutated values:
 * 	out_w := input_w - f_spatial_size + 1
 * 	out_h := input_h - f_spatial_size + 1
 *
 * clang-format on
 */

namespace test {
namespace data {

bool TestDataProvider::read(char const* const file) {
  std::cout << "Loading test data from: '" << file << "'" << std::endl;

  JsonValue value;
  JsonAllocator allocator;
  cnn_sr::utils::read_json_file(file, value, allocator, JSON_OBJECT);

  bool read_status = true;
  if (value.getTag() == JSON_OBJECT) {
    for (auto object : value) {
      //
      if (object->value.getTag() != JSON_OBJECT) continue;

      if (strcmp(object->key, "layer_1") == 0) {
        read_status &= read_layer_data(object->value, layer1_data);
      } else if (strcmp(object->key, "layer_2_data_set_1") == 0) {
        read_status &= read_layer_data(object->value, layer2_data_set1);
      } else if (strcmp(object->key, "layer_2_data_set_2") == 0) {
        read_status &= read_layer_data(object->value, layer2_data_set2);
      } else if (strcmp(object->key, "layer_3") == 0) {
        read_status &= read_layer_data(object->value, layer3_data);
      }
      //
    }
  }

  return read_status;
}

bool TestDataProvider::read_layer_data(const JsonValue& object,
                                       LayerData& data) {
  // ASSERT(object.getTag() == JSON_TAG_OBJECT);

  for (auto node : object) {
    JSON_READ_STR (node, data, name)
    JSON_READ_UINT(node, data, n_prev_filter_cnt)
    JSON_READ_UINT(node, data, f_spatial_size)
    JSON_READ_UINT(node, data, current_filter_count)
    JSON_READ_UINT(node, data, input_w)
    JSON_READ_UINT(node, data, input_h)
    JSON_READ_NUM_ARRAY(node, data, input)
    JSON_READ_NUM_ARRAY(node, data, output)
    JSON_READ_NUM_ARRAY(node, data, weights)
    JSON_READ_NUM_ARRAY(node, data, bias)
    JSON_READ_UINT(node, data, result_multiply)
    JSON_READ_UINT(node, data, preproces_mean)
  }

  return true;
}

//
}
}
