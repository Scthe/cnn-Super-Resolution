#include "pch.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>  // std::runtime_error
#include <ios>        // std::ios_base::failure
#include <dirent.h>   // list files in directory
#include <cstdlib>    // for string -> number conversion
#include <cstring>    // for strcmp/strlen when reading json
#
#include "json/gason.h"

namespace cnn_sr {
namespace utils {

///
/// Utils
///
void require(bool check, const char* msg) {
  if (!check) {
    throw std::runtime_error(msg);
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

///
/// File system
///
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

void list_files(const char* const path, std::vector<std::string>& target) {
  DIR* d;
  struct dirent* dir;
  d = opendir(path);
  if (d) {
    while ((dir = readdir(d)) != NULL) {
      target.push_back(dir->d_name);
      // if (dir->d_type == DT_REG) { // regular (non dirs/links)
      // printf("%s\n", dir->d_name);
      // }
      // printf("%s\n", dir->d_name);
    }

    closedir(d);
  }
}

///
/// Json utils
///
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

bool try_read_float(JsonNode& node, float& lhs, const char* key) {
  if (strcmp(node.key, key) == 0 && node.value.getTag() == JSON_NUMBER) {
    lhs = (float)node.value.toNumber();
    return true;
  }
  return false;
}

bool try_read_uint(JsonNode& node, unsigned int& lhs, const char* key) {
  if (strcmp(node.key, key) == 0 && node.value.getTag() == JSON_NUMBER) {
    lhs = (unsigned int)node.value.toNumber();
    return true;
  }
  return false;
}

bool try_read_vector(JsonNode& node, std::vector<float>& lhs, const char* key) {
  // std::cout << "key >> '" << node.key << "' vs '" << key
  // << "' == " << (strcmp(node.key, key)) << std::endl;
  if (strcmp(node.key, key) == 0 && node.value.getTag() == JSON_ARRAY) {
    auto arr_raw = node.value;
    for (auto val : arr_raw) {
      /* ASSERT(val->value.getTag() == JSON_NUMBER);*/
      float v = (float)val->value.toNumber();
      lhs.push_back(v);
    }
    return true;
  }
  return false;
}

bool try_read_string(JsonNode& node, std::string& lhs, const char* key) {
  if (strcmp(node.key, key) == 0 && node.value.getTag() == JSON_STRING) {
    lhs.assign(node.value.toString());
    return true;
  }
  return false;
}

///
/// Cmd line args parsing
///

ArgOption& ArgOption::help(const char* text) {
  _help.assign(text);
  return *this;
}

ArgOption& ArgOption::required() {
  _required = true;
  return *this;
}

Argparse::Argparse(const char* exec, const char* help)
    : _general_help(help), _exec_name(exec) {
  add_argument("help", "-h").help("Print this help");
}

ArgOption& Argparse::add_argument(const char* m1) {
  return add_argument(1, &m1);
}

ArgOption& Argparse::add_argument(const char* m1, const char* m2) {
  const char* tmp[2] = {m1, m2};
  return add_argument(2, tmp);
}

bool is_valid_arg(size_t len, const char* ms) {
  bool valid = len > 0;
  for (size_t i = 0; i < len; i++) {
    auto c = ms[i];
    valid &= isalpha(c) || (c == '-');
  }
  if (!valid) std::cout << "'" << ms << "' is not valid mnemonic" << std::endl;
  return valid;
}

ArgOption& Argparse::add_argument(size_t mlen, const char** ms) {
  _options.push_back(ArgOption());
  ArgOption& opt = _options.back();
  for (size_t i = 0; i < mlen; i++) {
    auto mnemonic = const_cast<char*>(ms[i]);
    // std::cout << mnemonic << std::endl;
    auto len = strlen(mnemonic);
    if (!is_valid_arg(len, mnemonic)) continue;

    char* name = nullptr;
    if (mnemonic[0] != '-')
      name = mnemonic;
    else if (len > 2 && mnemonic[0] == '-' && mnemonic[1] == '-')
      name = mnemonic + 2;
    // name = mnemonic;
    if (name) {
      opt._name.assign(name);
      // std::cout << "NAME: " << name << std::endl;
    }

    opt._mnemonics.push_back(std::string(mnemonic));
  }
  if (opt._mnemonics.empty())
    throw std::runtime_error("Argument does not have valid mnemonic");
  if (opt._name.empty())
    throw std::runtime_error(
        "Argument does not have valid name (at least one mnemonic should: "
        "not "
        "have '-' prefix or start with '--')");
  return opt;
}

bool Argparse::parse(size_t argc, char** argv) {
  _values.clear();
  for (size_t i = 1; i < argc; i++) {  // skip file name
    auto arg = argv[i];
    // std::cout << "[" << i << "]" << arg << std::endl;

    // find ArgOption
    ArgOption* opt = nullptr;
    for (auto& o : _options) {
      for (auto& mnemonic : o._mnemonics) {
        // std::cout << "\t" << arg << " vs " << mnemonic << std::endl;
        if (mnemonic.compare(arg) == 0) opt = &o;
      }
      if (opt) break;
    }
    if (!opt) {
      std::cout << "Unrecognised argument: '" << arg << "'" << std::endl;
      continue;
    }

    // read value
    bool has_value = opt->_mnemonics[0][0] == '-';
    if (!has_value) {
      _values.push_back(ArgValue(opt, ""));
    } else if (i + 1 < argc) {
      ++i;
      auto val = argv[i];
      _values.push_back(ArgValue(opt, std::string(val)));
    } else {
      std::cout << "Expected value for: '" << opt->_name << "'" << std::endl;
      continue;
    }

    // ArgValue& val = _values.back();
    // std::cout << "Arg [" << val.first->_name << "] '" << val.second << "'"
    // << std::endl;
  }

  auto only_help = has_arg("help");
  if (only_help) {
    print_help();
    return false;
  }

  // check required args - all should be provided
  for (auto& o : _options) {
    if (!o._required) continue;
    bool provided = false;
    for (auto& val : _values) {
      if (val.first == &o) provided |= true;
    }
    if (!provided) {
      char buf[255];
      snprintf(buf, 255, "Value not provided for argument: '%s'",
               o._name.c_str());
      throw std::runtime_error(buf);
    }
  }

  return true;
}

void print_argument(std::ostream& os, ArgOption& opt) {
  // print mnemonics
  // bool has_value = opt._name.compare(0, 2, "--") == 0;
  bool has_value = opt._mnemonics[0][0] == '-';
  std::string name_upper = opt._name;
  for (auto& c : name_upper) c = toupper(c);

  for (size_t i = 0; i < opt._mnemonics.size(); i++) {
    auto& mnemonic = opt._mnemonics[i];
    os << "  " << mnemonic;
    if (has_value) os << " " << name_upper;

    if (i != opt._mnemonics.size() - 1) os << ", ";
  }

  // print help
  os << std::endl
     << "                 "  // this makes things easier
     << opt._help;
}

void Argparse::print_help() {
  std::ostream& os = std::cout;
  os << _exec_name;

  // print args line
  for (auto& opt : _options) {
    os << " ";
    if (!opt._required) os << "[";
    // if (!opt._name.empty())
    // os << opt._name;
    // else
    // os << opt._mnemonics[0];
    os << opt._mnemonics.back();
    if (!opt._required) os << "]";
  }

  // print description
  os << std::endl
     << std::endl
     << _general_help << std::endl
     << std::endl;

  // arguments
  os << "arguments:" << std::endl;
  for (auto& opt : _options) {
    print_argument(os, opt);
    os << std::endl;
  }
}

Argparse::ArgValue* Argparse::get_value(const char* arg_name) {
  Argparse::ArgValue* v = nullptr;
  for (auto& val : _values) {
    if (val.first->_name.compare(arg_name) == 0) {
      v = &val;
      break;
    }
  }
  return v;
}

bool Argparse::has_arg(const char* arg_name) {
  return get_value(arg_name) != nullptr;
}

const char* Argparse::value(const char* arg_name) {
  auto v = get_value(arg_name);
  return v != nullptr ? v->second.c_str() : nullptr;
}

void Argparse::value(const char* arg_name, size_t& val) {
  auto v = get_value(arg_name);
  if (!v) return;
  val = atoi(v->second.c_str());  // ignore value overflow..
}

//
}
}
