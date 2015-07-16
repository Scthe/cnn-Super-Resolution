#include "Utils.hpp"

#include <string>
#include <fstream>
#include <stdexcept>  // std::runtime_error
#include <ios>        // std::ios_base::failure
#include "json/gason.h"

namespace cnn_sr {
namespace utils {

void get_file_content(const char* const filename, std::stringstream& sstr) {
  // TODO use to load kernel file too
  std::fstream file(filename);
  if (!file.is_open()) {
    throw std::ios_base::failure("File not found");
  }

  std::string line;
  while (file.good()) {
    getline(file, line);
    sstr << line;
  }
}

void dump_vector(std::ostream& os, std::vector<float>& data,
                 const char* line_prefix, size_t per_line,
                 bool add_line_numbers) {
  auto len = data.size();
  size_t lines;
  if (per_line == 0) {
    lines = 1;
    per_line = len;
  } else {
    lines = len / per_line;
  }

  line_prefix = line_prefix ? line_prefix : "";

  for (size_t i = 0; i < lines; i++) {
    os << line_prefix;
    if (add_line_numbers) os << "[" << i << "] ";

    for (size_t j = 0; j < per_line; j++) {
      size_t idx = i * per_line + j;
      if (idx < len) os << data[idx];
      if (j + 1 < per_line) os << ", ";
    }
    if (i + 1 < lines) os << std::endl;
  }
}

void read_json_file(const char* const file, JsonValue& value,
                    JsonAllocator& allocator, std::string& file_content,
                    int root_type) {
  if (strlen(file) > 250) {
    throw IOException("Filepath is too long");
  }

  std::stringstream sstr;
  get_file_content(file, sstr);
  file_content = sstr.str();
  char* source = const_cast<char*>(file_content.c_str());

  char* endptr;
  auto status = jsonParse(source, &endptr, &value, allocator);
  if (status != JSON_OK) {
    char buf[255];
    snprintf(buf, 255, "Json parsing error: %s in: '%-20s'",
             jsonStrError(status), endptr);
    throw IOException(buf);
  }

  if (value.getTag() != root_type) {
    throw std::runtime_error("Expected root of JSON file had invalid type");
  }
}

//
}
}
