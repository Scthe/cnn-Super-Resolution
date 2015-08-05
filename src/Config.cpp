#include "Config.hpp"
#include <cmath>    // for std::abs
#include <cstring>  // for strcmp when reading json

#include "json/gason.h"

namespace cnn_sr {
using namespace utils;

const char* const parameters_keys[3] = {"parameters_distribution_1",
                                        "parameters_distribution_2",
                                        "parameters_distribution_3"};

ParametersDistribution::ParametersDistribution(float mean_w, float mean_b,
                                               float sd_w, float sd_b)
    : mean_w(mean_w), sd_w(sd_w), mean_b(mean_b), sd_b(sd_b) {}

///
/// Config
///
Config::Config(size_t n1, size_t n2,                                       //
               size_t f1, size_t f2, size_t f3,                            //
               float momentum, float weight_decay, float* learning_rates,  //
               ParametersDistribution pd1,                                 //
               ParametersDistribution pd2,                                 //
               ParametersDistribution pd3,                                 //
               const char* const parameters_file)
    : n1(n1),
      n2(n2),
      f1(f1),
      f2(f2),
      f3(f3),
      momentum(momentum),
      weight_decay_parameter(weight_decay),
      parameters_file(parameters_file),
      params_distr_1(pd1),
      params_distr_2(pd2),
      params_distr_3(pd3) {
  for (size_t i = 0; i < 3; i++) {
    this->learning_rate[i] = learning_rates[i];
  }
}

size_t Config::total_padding() const { return f1 + f2 + f3 - 3; }

void Config::validate(Config& config) {
  // spatial size works best if is odd number
  utils::require(is_odd(config.f1), "f1 should be odd");
  utils::require(is_odd(config.f2), "f2 should be odd");
  utils::require(is_odd(config.f3), "f3 should be odd");
  // both filter count and spatial size cannot be 0
  utils::require(config.n1 > 0, "n1 should be >0");
  utils::require(config.n2 > 0, "n2 should be >0");
  utils::require(config.f1 > 0, "f1 should be >0");
  utils::require(config.f2 > 0, "f2 should be >0");
  utils::require(config.f3 > 0, "f3 should be >0");

  utils::require(config.f3 > 0, "f3 should be >0");
  utils::require(config.weight_decay_parameter >= 0,
                 "weight_decay should be >0");
  utils::require(config.learning_rate[0] > 0 && config.learning_rate[1] > 0 &&
                     config.learning_rate[2] > 0,
                 "All learning rates should be >0");

  // ParametersDistribution
  ParametersDistribution* pd_arr[3] = {&config.params_distr_1,  //
                                       &config.params_distr_2,  //
                                       &config.params_distr_3};
  for (auto i = 0; i < 3; i++) {
    auto pd = pd_arr[i];
    utils::require(pd->sd_w > 0, "std dev. for weights should be > 0");
    utils::require(pd->sd_b >= 0, "std dev. for bias should be >= 0");
  }
}

///
/// ConfigReader
///

struct ConfigHelper {
  size_t n1, n2, f1, f2, f3;
  float momentum, weight_decay, lr1, lr2, lr3;
  std::string parameters_file = "";
  std::vector<float> learning_rates;
};

void fix_params_distribution(ParametersDistribution& d) {
  d.mean_w = std::abs(d.mean_w);
  d.mean_b = std::abs(d.mean_b);
  d.sd_w = std::abs(d.sd_w);
  d.sd_b = std::abs(d.sd_b);
}

void load_parameters_distr(JsonNode* node, ParametersDistribution& data) {
  for (auto subnode : node->value) {
    utils::try_read_float(*subnode, data.mean_w, "mean_w");
    utils::try_read_float(*subnode, data.mean_b, "mean_b");
    utils::try_read_float(*subnode, data.sd_w, "std_deviation_w");
    utils::try_read_float(*subnode, data.sd_b, "std_deviation_b");
  }
}

Config ConfigReader::read(const char* const file) {
  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  utils::read_json_file(file, value, allocator, source, JSON_OBJECT);

  ConfigHelper cfg_h;
  ParametersDistribution pd1, pd2, pd3;
  for (auto node : value) {
    auto key = node->key;
    utils::try_read_uint(*node, cfg_h.n1, "n1");
    utils::try_read_uint(*node, cfg_h.n2, "n2");
    utils::try_read_uint(*node, cfg_h.f1, "f1");
    utils::try_read_uint(*node, cfg_h.f2, "f2");
    utils::try_read_uint(*node, cfg_h.f3, "f3");
    utils::try_read_float(*node, cfg_h.momentum, "momentum");
    utils::try_read_float(*node, cfg_h.weight_decay, "weight_decay_parameter");
    utils::try_read_string(*node, cfg_h.parameters_file, "parameters_file");
    utils::try_read_vector(*node, cfg_h.learning_rates, "learning_rates");

    if (strcmp(key, parameters_keys[0]) == 0) {
      load_parameters_distr(node, pd1);
    } else if (strcmp(key, parameters_keys[1]) == 0) {
      load_parameters_distr(node, pd2);
    } else if (strcmp(key, parameters_keys[2]) == 0) {
      load_parameters_distr(node, pd3);
    }
  }

  fix_params_distribution(pd1);
  fix_params_distribution(pd2);
  fix_params_distribution(pd3);
  utils::require(cfg_h.learning_rates.size() == 3,
                 "Expected 3 learning rates (one per layer) to be provided");

  Config cfg(cfg_h.n1, cfg_h.n2,            //
             cfg_h.f1, cfg_h.f2, cfg_h.f3,  //
             cfg_h.momentum, cfg_h.weight_decay,
             &cfg_h.learning_rates[0],  //
             pd1, pd2, pd3,             //
             cfg_h.parameters_file.c_str());
  Config::validate(cfg);

  return cfg;
}
}

std::ostream& operator<<(std::ostream& os,
                         const cnn_sr::ParametersDistribution& pd) {
  /* clang-format off */
  os << "{ weights(" << pd.mean_w << ", " << pd.sd_w
     << "), bias("   << pd.mean_b << ", " << pd.sd_b << ")}";
  /* clang-format on */
  return os;
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::Config& cfg) {
  /* clang-format off */
  os << "Config {" << std::endl
     << "  parameters file: '" << cfg.parameters_file << "'" << std::endl
     << "  momentum: " << cfg.momentum << std::endl
     << "  learning rates: { " << cfg.learning_rate[0] << ", "
                               << cfg.learning_rate[1] << ", "
                               << cfg.learning_rate[2] << "}" << std::endl
     << "  layer 1: " << cfg.n1 << " filters, " << cfg.f1 << " spatial size" << std::endl
     << "  layer 2: " << cfg.n2 << " filters, " << cfg.f2 << " spatial size" << std::endl
     << "  layer 3: " << cfg.f3 << " spatial size" << std::endl
     << "  parameters dist. 1 " << cfg.params_distr_1 << std::endl
     << "  parameters dist. 2 " << cfg.params_distr_2 << std::endl
     << "  parameters dist. 3 " << cfg.params_distr_3 << "}" << std::endl;
  /* clang-format on */
  return os;
}
