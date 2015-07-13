#include "Config.hpp"
#include <iostream>   // for std::cout
#include <stdexcept>  // std::runtime_error
#include <cmath>      // for std::abs
#include <array>
#include <sstream>

#include "json/gason.h"
#include "Utils.hpp"

namespace cnn_sr {
using namespace utils;

const char* const parameters_keys[3] = {"parameters_distribution_1",
                                        "parameters_distribution_2",
                                        "parameters_distribution_3"};

ParametersDistribution::ParametersDistribution(float mean_w, float mean_b,
                                               float sd_w, float sd_b)
    : mean_w(mean_w), sd_w(sd_w), mean_b(mean_b), sd_b(sd_b) {}

///
/// Vaildation macros. @see Config::validate
///
#define GT_ZERO(VALUE)                                                  \
  if (VALUE <= 0) {                                                     \
    is_correct = false;                                                 \
    err_stream << "Value of " << #VALUE                                 \
               << "should be greater then zero, meanwhile is " << VALUE \
               << std::endl;                                            \
  }

#define GTE_ZERO(VALUE)                                                        \
  if (VALUE < 0) {                                                             \
    is_correct = false;                                                        \
    err_stream << "Value of " << #VALUE                                        \
               << "should be greater/equall then zero, meanwhile is " << VALUE \
               << std::endl;                                                   \
  }

#define IS_ODD(VALUE)                                                \
  if (!is_odd(VALUE)) {                                              \
    is_correct = false;                                              \
    err_stream << "Value of " << #VALUE                              \
               << " should be an odd number, meanwhile is " << VALUE \
               << std::endl;                                         \
  }

///
/// Config
///
Config::Config(size_t n1, size_t n2,             //
               size_t f1, size_t f2, size_t f3,  //
               ParametersDistribution pd1,       //
               ParametersDistribution pd2,       //
               ParametersDistribution pd3,       //
               const char* const parameters_file)
    : n1(n1),
      n2(n2),
      f1(f1),
      f2(f2),
      f3(f3),
      params_distr_1(pd1),
      params_distr_2(pd2),
      params_distr_3(pd3) {
  auto len = strlen(parameters_file);
  this->parameters_file = new char[len + 1];
  strncpy(this->parameters_file, parameters_file, len);
  this->parameters_file[len] = '\0';
}

Config::~Config() { delete[] this->parameters_file; }

void Config::validate(Config& config) {
  std::stringstream err_stream;
  bool is_correct = true;
  // spatial size works best if is odd number
  IS_ODD(config.f1)
  IS_ODD(config.f2)
  IS_ODD(config.f3)
  // both filter count and spatial size cannot be 0
  GT_ZERO(config.n1)
  GT_ZERO(config.n2)
  GT_ZERO(config.f1)
  GT_ZERO(config.f2)
  GT_ZERO(config.f3)

  // ParametersDistribution
  std::array<ParametersDistribution*, 3> pds = {{&config.params_distr_1,  //
                                                 &config.params_distr_2,  //
                                                 &config.params_distr_3}};
  for (auto e = begin(pds); e != end(pds); ++e) {
    auto params_distr = *e;
    GTE_ZERO(params_distr->mean_w)
    GTE_ZERO(params_distr->mean_b)
    GTE_ZERO(params_distr->sd_w)
    GTE_ZERO(params_distr->sd_b)
    if (params_distr->sd_w == 0.0f) {
      is_correct = false;
      err_stream << "Value of standard deviation for weights should not be 0"
                 << std::endl;
    }
  }

  if (!is_correct) {
    throw std::runtime_error(err_stream.str());
  }
}

///
/// ConfigReader
///

struct ConfigHelper {
  size_t n1, n2, f1, f2, f3;
  const char* parameters_file = nullptr;
};

void fix_params_dist(ParametersDistribution& d) {
  d.mean_w = std::abs(d.mean_w);
  d.mean_b = std::abs(d.mean_b);
  d.sd_w = std::abs(d.sd_w);
  d.sd_b = std::abs(d.sd_b);
}

void load_parameters_distr(JsonNode* node, ParametersDistribution& data) {
  for (auto subnode : node->value) {
    JSON_READ_FLOAT(subnode, data.mean_w, "mean_w")
    JSON_READ_FLOAT(subnode, data.mean_b, "mean_b")
    JSON_READ_FLOAT(subnode, data.sd_w, "std_deviation_w")
    JSON_READ_FLOAT(subnode, data.sd_b, "std_deviation_b")
  }
}

Config ConfigReader::read(const char* const file) {
  std::cout << "Loading config from: '" << file << "'" << std::endl;

  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  utils::read_json_file(file, value, allocator, source, JSON_OBJECT);

  ConfigHelper cfg_h;
  ParametersDistribution pd1, pd2, pd3;
  for (auto node : value) {
    auto key = node->key;
    JSON_READ_UINT(node, cfg_h, n1)
    JSON_READ_UINT(node, cfg_h, n2)
    JSON_READ_UINT(node, cfg_h, f1)
    JSON_READ_UINT(node, cfg_h, f2)
    JSON_READ_UINT(node, cfg_h, f3)
    JSON_READ_STR(node, cfg_h, parameters_file)
    if (strcmp(key, parameters_keys[0]) == 0) {
      load_parameters_distr(node, pd1);
    } else if (strcmp(key, parameters_keys[1]) == 0) {
      load_parameters_distr(node, pd2);
    } else if (strcmp(key, parameters_keys[2]) == 0) {
      load_parameters_distr(node, pd3);
    }
  }

  fix_params_dist(pd1);
  fix_params_dist(pd2);
  fix_params_dist(pd3);

  Config cfg(cfg_h.n1, cfg_h.n2,            //
             cfg_h.f1, cfg_h.f2, cfg_h.f3,  //
             pd1, pd2, pd3,                 //
             cfg_h.parameters_file);
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
     << "  layer 1: " << cfg.n1 << " filters, " << cfg.f1 << " spatial size" << std::endl
     << "  layer 2: " << cfg.n2 << " filters, " << cfg.f2 << " spatial size" << std::endl
     << "  layer 3: " << cfg.f3 << " spatial size" << std::endl
     << "  parameters file: '" << cfg.parameters_file << "'" << std::endl
     << "  parameters dist. 1 " << cfg.params_distr_1 << std::endl
     << "  parameters dist. 2 " << cfg.params_distr_2 << std::endl
     << "  parameters dist. 3 " << cfg.params_distr_3 << "}" << std::endl;
  /* clang-format on */
  return os;
}
