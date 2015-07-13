#ifndef CONFIG__H
#define CONFIG__H

#include <cstddef>  // for size_t
#include <ostream>  // for std::ostream& operator<<(..)

union JsonValue;

namespace cnn_sr {

struct ParametersDistribution {
  ParametersDistribution() {}
  ParametersDistribution(float, float, float, float);

  float mean_w = 0.01f, sd_w = 0.01f;
  float mean_b = 0.0f, sd_b = 0.0f;
};

struct Config {
  Config(size_t, size_t,          //
         size_t, size_t, size_t,  //
         float, float, float*,    //
         ParametersDistribution, ParametersDistribution, ParametersDistribution,
         const char* const = nullptr);
  ~Config();

  static void validate(Config&);

  // core parameters
  const size_t n1, n2;
  const size_t f1, f2, f3;
  const float momentum, weight_decay_parameter;
  float learning_rate[3];
  char* parameters_file = nullptr;

  // random parameters(weights/biases)
  ParametersDistribution params_distr_1;
  ParametersDistribution params_distr_2;
  ParametersDistribution params_distr_3;
};

class ConfigReader {
 public:
  Config read(const char* const);
};
}

std::ostream& operator<<(std::ostream&, const cnn_sr::Config&);

#endif /* CONFIG__H   */
