#ifndef __CONFIG__H
#define __CONFIG__H

#include <string>

union JsonValue;

namespace cnn_sr {

struct Config {
  Config(const char* const);

  size_t n1, n2;  // TODO make const
  size_t f1, f2, f3;

  // TODO for now only small.jpg, large.jpg
  std::string parameters_file;
  const std::string source_file;
};

class ConfigReader {
 public:
  // typedef void (*Reader)(JsonValue&, Config&);
  Config read(const char* const);

 private:
  void validate(const Config&);
};
}

std::ostream& operator<<(std::ostream&, const cnn_sr::Config&);

#endif /* __CONFIG__H   */
