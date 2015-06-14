#include "Config.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <ios>  // std::ios_base::failure

#include "json/gason.h"

typedef std::ios_base::failure IOException;

// TODO move to utils header
void getFileContent(const char* const filename, std::stringstream& sstr) {
  // TODO use to load kernel file too
  std::fstream file(filename);
  if (!file.is_open()) {
    throw IOException("File not found");
  }

  std::string line;
  while (file.good()) {
    getline(file, line);
    sstr << line;
  }
}

#define READ_INT(NODE, OBJECT, PROP_NAME)                          \
  if (strcmp(NODE->key, #PROP_NAME) == 0 &&                        \
      NODE->value.getTag() == JSON_NUMBER) {                       \
    OBJECT.PROP_NAME = (unsigned int)NODE->value.toNumber();       \
    /*std::cout << "INT: " << NODE->key << " = " << OBJECT.PROP_NAME \
              << std::endl;*/                                        \
  }

bool is_odd(size_t x) { return (x & 1) != 0; }

namespace cnn_sr {

Config::Config(const char* const src_file)
    : n1(0), n2(0), f1(0), f2(0), f3(0), source_file(src_file) {}

Config ConfigReader::read(const char* const file) {
  std::cout << "Loading config from: '" << file << "'" << std::endl;

  if (strlen(file) > 30) {
    throw IOException("Filepath is too long");
  }

  std::stringstream sstr;
  getFileContent(file, sstr);

  const std::string& tmp = sstr.str();
  char* source = const_cast<char*>(tmp.c_str());

  char* endptr;
  JsonValue value;
  JsonAllocator allocator;
  auto status = jsonParse(source, &endptr, &value, allocator);
  if (status != JSON_OK) {
    char buf[255];
    sprintf(buf, "Json parsing error: %s in: '%-20s'", jsonStrError(status),
            endptr);
    throw IOException(buf);
  }

  Config cfg(file);
  if (value.getTag() == JSON_OBJECT) {
    for (auto node : value) {
      READ_INT(node, cfg, n1)
      READ_INT(node, cfg, n2)
      READ_INT(node, cfg, f1)
      READ_INT(node, cfg, f2)
      READ_INT(node, cfg, f3)
    }
  }

  if (!validate(cfg)) {
    throw IOException("All spatial sizes ('fX' values) should be odd");
  }

  return cfg;
}

bool ConfigReader::validate(const Config& cfg) {
  return is_odd(cfg.f1) && is_odd(cfg.f2) && is_odd(cfg.f3);
}
}

std::ostream& operator<<(std::ostream& os, const cnn_sr::Config& cfg) {
  // os << platform_info.vendor << "::" << platform_info.name << ", version "
  //  << platform_info.version;
  os << "Config from file: " << cfg.source_file << std::endl
     << "  layer 1: " << cfg.n1 << " filters, " << cfg.f1 << " spatial size"
     << std::endl
     << "  layer 2: " << cfg.n2 << " filters, " << cfg.f2 << " spatial size"
     << std::endl
     << "  layer 3: " << cfg.f3 << " spatial size" << std::endl;
  return os;
}
