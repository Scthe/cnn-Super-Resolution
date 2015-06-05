#ifndef __TEST_DATA_PROVIDER_H
#define __TEST_DATA_PROVIDER_H

#include "json/gason.h"
#include <vector>

namespace test {
namespace data {

struct Layer1Data {
  size_t n1 = 0, f1 = 0;
  int input_w = 0, input_h = 0;
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<float> bias;
};

struct Layer2Data {
  size_t n2 = 0, f2 = 0;
  std::vector<float> input;
  std::vector<float> output;
  std::vector<float> weights;
  std::vector<float> bias;
};

class TestDataProvider {
 public:
  bool read(char const* const file);

 public:
  Layer1Data layer1_data;
  Layer2Data layer2_data_set1;
  Layer2Data layer2_data_set2;

 private:
  bool read_layer_1_data(const JsonValue&, Layer1Data&);
  bool read_layer_2_data(const JsonValue&, Layer2Data&);
};
}
}

#endif /* __TEST_DATA_PROVIDER_H   */
