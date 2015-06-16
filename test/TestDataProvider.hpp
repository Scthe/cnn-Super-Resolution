#ifndef __TEST_DATA_PROVIDER_H
#define __TEST_DATA_PROVIDER_H

#include <vector>
#include <string>

union JsonValue;

namespace test {
namespace data {

struct LayerData {
  std::string name;
  size_t n_prev_filter_cnt,
         current_filter_count,
         f_spatial_size,
         input_w, input_h;
  std::vector<float> input;
  std::vector<float> output;
  std::vector<float> weights;
  std::vector<float> bias;
  // optional:
  int result_multiply = 0;
  int preproces_mean = 0; // subtract mean from input
};

class TestDataProvider {
 public:
  bool read(char const* const file);

 public:
  LayerData layer1_data;
  LayerData layer2_data_set1;
  LayerData layer2_data_set2;
  LayerData layer3_data;

 private:
  bool read_layer_data(const JsonValue&, LayerData&);
};
}
}

#endif /* __TEST_DATA_PROVIDER_H   */
