#include "Utils.hpp"

#include <string>
#include <fstream>
#include <ios>  // std::ios_base::failure

namespace cnn_sr {
namespace utils {

void getFileContent(const char* const filename, std::stringstream& sstr) {
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

//
}
}
