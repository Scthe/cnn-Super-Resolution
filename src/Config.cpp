#include "Config.hpp"
#include <cstdio>  // snprintf
#include <ios>     // std::ios_base::failure
#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <sstream>

#include "json/gason.h"
#include "Utils.hpp"

namespace cnn_sr {
using namespace utils;

Config::Config(const char* const src_file)
    : n1(0), n2(0), f1(0), f2(0), f3(0), source_file(src_file) {}

Config ConfigReader::read(const char* const file) {
  std::cout << "Loading config from: '" << file << "'" << std::endl;

  JsonValue value;
  JsonAllocator allocator;
  std::string source;
  utils::read_json_file(file, value, allocator, source, JSON_OBJECT);

  Config cfg(file);
  for (auto node : value) {
    JSON_READ_UINT(node, cfg, n1)
    JSON_READ_UINT(node, cfg, n2)
    JSON_READ_UINT(node, cfg, f1)
    JSON_READ_UINT(node, cfg, f2)
    JSON_READ_UINT(node, cfg, f3)
  }

  validate(cfg);

  return cfg;
}

void ConfigReader::validate(const Config& cfg) {
#define GT_ZERO(PROP_NAME)                                     \
  if (cfg.PROP_NAME == 0) {                                    \
    is_correct = false;                                        \
    err_stream << "Value of " << #PROP_NAME                    \
               << "should be greater then zero, meanwhile is " \
               << cfg.PROP_NAME << std::endl;                  \
  }

#define IS_ODD(PROP_NAME)                                                    \
  if (!is_odd(cfg.PROP_NAME)) {                                              \
    is_correct = false;                                                      \
    err_stream << "Value of " << #PROP_NAME                                  \
               << " should be an odd number, meanwhile is " << cfg.PROP_NAME \
               << std::endl;                                                 \
  }

  std::stringstream err_stream;
  bool is_correct = true;
  GT_ZERO(n1)
  GT_ZERO(n2)
  GT_ZERO(f1)
  GT_ZERO(f2)
  GT_ZERO(f3)

  IS_ODD(f1)
  IS_ODD(f2)
  IS_ODD(f3)

  if (!is_correct) {
    throw std::runtime_error(err_stream.str());
  }

#undef GT_ZERO
#undef IS_ODD
}
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::Config& cfg) {
  os << "Config from file: '" << cfg.source_file << std::endl
     << "'  layer 1: " << cfg.n1 << " filters, " << cfg.f1 << " spatial size"
     << std::endl
     << "  layer 2: " << cfg.n2 << " filters, " << cfg.f2 << " spatial size"
     << std::endl
     << "  layer 3: " << cfg.f3 << " spatial size" << std::endl;
  return os;
}
