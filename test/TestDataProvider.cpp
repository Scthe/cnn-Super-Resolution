#include "TestDataProvider.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>

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


bool getFileContent(const char* const filename, std::stringstream& sstr) {
  // TODO use to load kernel file too
  std::fstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (file.good()) {
    getline(file, line);
    // LOGD << line;
    sstr << line;
  }
  return true;
}

namespace test {
namespace data {

bool TestDataProvider::read(char const* const file) {
  std::cout << "Loading test data from: '" << file << "'" << std::endl;

  if (strlen(file) > 230) {
    std::cout << "test data filepath is too long (max 230 chars)" << std::endl;
    return false;
  }

  std::stringstream sstr;
  bool ok = getFileContent(file, sstr);
  if (!ok) {
    std::cout << "file not found" << std::endl;
    return false;
  }

  const std::string& tmp = sstr.str();
  const char* source_ = tmp.c_str();
  char* source = const_cast<char*>(source_);

  char* endptr;
  JsonValue value;
  JsonAllocator allocator;
  auto status = jsonParse(source, &endptr, &value, allocator);
  if (status != JSON_OK) {
    char buf[255];
    sprintf(buf, "Json parsing error at %zd, status: %d", endptr - source,
            (int)status);
    std::cout << buf << std::endl;
    return false;
  }

  bool read_status = true;
  if (value.getTag() == JSON_OBJECT) {
    for (auto object : value) {
      //
      if (object->value.getTag() != JSON_OBJECT) continue;

      if (strcmp(object->key, "layer_1") == 0) {
        read_status &= read_layer_data(object->value, layer1_data);
      } else if (strcmp(object->key, "layer_3") == 0) {
        read_status &= read_layer_data(object->value, layer3_data);
      }  else if (strcmp(object->key, "layer_2") == 0) {
         for (auto node : object->value) {
           if (strcmp(node->key, "data_set_1") == 0 &&
               node->value.getTag() == JSON_OBJECT) {
             read_status &= read_layer_data(node->value, layer2_data_set1);
           } else if (strcmp(node->key, "data_set_2") == 0 &&
                      node->value.getTag() == JSON_OBJECT) {
             read_status &= read_layer_data(node->value, layer2_data_set2);
           }
         }
       }
      //
    }
  }

  return read_status;
}

#define READ_INT(PROP_NAME)                                                    \
  if (strcmp(node->key, #PROP_NAME) == 0 &&                                    \
      node->value.getTag() == JSON_NUMBER) {                                   \
    /* TODO ASSERT(node.getTag() == JSON_NUMBER);*/                            \
    data.PROP_NAME = (unsigned int)node->value.toNumber();                     \
    std::cout << "INT: " << node->key << " = " << data.PROP_NAME << std::endl; \
    read_##PROP_NAME = true;                                                   \
  }

#define READ_ARRAY(PROP_NAME)                          \
  if (strcmp(node->key, #PROP_NAME) == 0 &&            \
      node->value.getTag() == JSON_ARRAY) {            \
    auto arr_raw = node->value;                        \
    std::cout << "ARRAY: " << node->key << std::endl;  \
    for (auto val : arr_raw) {                         \
      /* ASSERT(val->value.getTag() == JSON_NUMBER);*/ \
      double num = val->value.toNumber();              \
      data.PROP_NAME.push_back(num);                   \
    }                                                  \
    read_##PROP_NAME = true;                           \
  }

bool TestDataProvider::read_layer_data(const JsonValue& object,
                                       LayerData& data) {
  // ASSERT(object.getTag() == JSON_TAG_OBJECT);

  /* clang-format off */
  bool read_n_prev_filter_cnt = false,
       read_current_filter_count = false,
       read_f_spatial_size = false,
       read_input_w = false, read_input_h = false,
       // vectors:
       read_input = false,
       read_output = false,
       read_weights = false,
       read_bias = false;
   bool read_result_multiply = true; // optional
   bool read_preproces_mean = true; // optional
  /* clang-format on */

  for (auto node : object) {
    // std::cout << i->key << std::endl;
    READ_INT(n_prev_filter_cnt)
    READ_INT(f_spatial_size)
    READ_INT(current_filter_count)
    READ_INT(input_w)
    READ_INT(input_h)
    READ_ARRAY(input)
    READ_ARRAY(output)
    READ_ARRAY(weights)
    READ_ARRAY(bias)
    READ_INT(result_multiply)
    READ_INT(preproces_mean)
  }

// TODO throw to not allow to run test with not valid data definitions
#define ASSERT_READ(PROP_NAME)                                            \
  if (!read_##PROP_NAME) {                                                \
    std::cout << "Expected to read: '" << #PROP_NAME << "'" << std::endl; \
    return false;                                                         \
  }

  ASSERT_READ(n_prev_filter_cnt);
  ASSERT_READ(current_filter_count);
  ASSERT_READ(f_spatial_size);
  ASSERT_READ(input_w);
  ASSERT_READ(input_h);
  ASSERT_READ(input);
  ASSERT_READ(output);
  ASSERT_READ(weights);
  ASSERT_READ(bias);

#undef ASSERT_READ

  return true;
}

//
}
}
