#include "Config.hpp"
#include <cstdio>  // snprintf
#include <ios>     // std::ios_base::failure
#include <iostream>

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
  utils::read_json_file(file, value, allocator, JSON_OBJECT);

  Config cfg(file);
  for (auto node : value) {
    JSON_READ_UINT(node, cfg, n1)
    JSON_READ_UINT(node, cfg, n2)
    JSON_READ_UINT(node, cfg, f1)
    JSON_READ_UINT(node, cfg, f2)
    JSON_READ_UINT(node, cfg, f3)
  }

  if (!validate(cfg)) {
    throw IOException("All spatial sizes ('fX' values) should be odd");
  }

  return cfg;
}

bool ConfigReader::validate(const Config& cfg) {
  // TODO check all values != 0
  return is_odd(cfg.f1) && is_odd(cfg.f2) && is_odd(cfg.f3);
}
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::Config& cfg) {
  os << "Config from file: " << cfg.source_file << std::endl
     << "  layer 1: " << cfg.n1 << " filters, " << cfg.f1 << " spatial size"
     << std::endl
     << "  layer 2: " << cfg.n2 << " filters, " << cfg.f2 << " spatial size"
     << std::endl
     << "  layer 3: " << cfg.f3 << " spatial size" << std::endl;
  return os;
}
