#include "TestDataProvider.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>

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
        read_status &= read_layer_1_data(object->value, layer1_data);
      } else if (strcmp(object->key, "layer_2") == 0) {
        for (auto node : object->value) {
          if (strcmp(node->key, "data_set_1") == 0 &&
              node->value.getTag() == JSON_OBJECT) {
            read_status &= read_layer_2_data(node->value, layer2_data);
          }
        }
      }
      //
    }
  }

  return read_status;
}

#define READ_INT(NAME)                                                        \
  if (strcmp(node->key, #NAME) == 0 && node->value.getTag() == JSON_NUMBER) { \
    /* TODO ASSERT(node.getTag() == JSON_NUMBER);*/                           \
    data.NAME = (int)node->value.toNumber();                                  \
    std::cout << "INT: " << node->key << " = " << data.NAME << std::endl;     \
  }

#define READ_ARRAY(NAME)                                                     \
  if (strcmp(node->key, #NAME) == 0 && node->value.getTag() == JSON_ARRAY) { \
    auto arr_raw = node->value;                                              \
    std::cout << "ARRAY: " << node->key << std::endl;                        \
    for (auto val : arr_raw) {                                               \
      /* ASSERT(val->value.getTag() == JSON_NUMBER);*/                       \
      double num = val->value.toNumber();                                    \
      data.NAME.push_back(num);                                              \
    }                                                                        \
  }

bool TestDataProvider::read_layer_1_data(const JsonValue& object,
                                         Layer1Data& data) {
  // ASSERT(object.getTag() == JSON_TAG_OBJECT);
  for (auto node : object) {
    // std::cout << i->key << std::endl;
    READ_INT(n1)
    READ_INT(f1)
    READ_INT(input_h)
    READ_INT(input_w)
    READ_ARRAY(input)
    READ_ARRAY(weights)
    READ_ARRAY(bias)
  }
  return true;
}

bool TestDataProvider::read_layer_2_data(const JsonValue& object,
                                         Layer2Data& data) {
  // ASSERT(object.getTag() == JSON_TAG_OBJECT);
  for (auto node : object) {
    // std::cout << i->key << std::endl;
    READ_INT(n2)
    READ_INT(f2)
    READ_ARRAY(input)
    READ_ARRAY(output)
    READ_ARRAY(weights)
    READ_ARRAY(bias)
  }
  return true;
}

//
}
}
