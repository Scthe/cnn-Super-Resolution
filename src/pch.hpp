#ifndef PCH_H
#define PCH_H

#include <string>
#include <vector>
// #include <cstddef>  // for size_t

// TODO use during compilation

///
/// forward declarations
///
/* clang-format off */
namespace cnn_sr {
  struct ParametersDistribution;
  struct Config;
  class ConfigReader;
  class DataPipeline;
  class ConfigBasedDataPipeline;
  struct LayerData;
  struct CnnLayerGpuAllocationPool;
}

namespace opencl {
  class Kernel;
  typedef size_t MemoryHandle;
  class Context;

  namespace utils {
    struct ImageData;
  }
}
/* clang-format on */

typedef struct _cl_event* cl_event;

union JsonValue;
struct JsonNode;
class JsonAllocator;

///
/// Utils
///
namespace cnn_sr {
namespace utils {

void require(bool, const char*);

void dump_vector(std::ostream&, std::vector<float>&,
                 const char* line_prefix = nullptr, size_t per_line = 0,
                 bool add_line_numbers = false);

template <typename T>
inline bool is_odd(T x) {
  return (x & 1) != 0;
}

template <typename T>
inline bool is_even(T x) {
  return !is_odd(x);
}

///
/// Utils - macros
///
#define STRINGIFY2(s) #s
#define STRINGIFY(s) STRINGIFY2(s)

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

///
/// File system
///
#define IOException std::ios_base::failure

void get_file_content(const char* const, std::stringstream&);

void list_files(const char* const, std::vector<std::string>&);

///
/// Json utils
///
/** NOTE: we need to hold file content in some persistent place, since the
 * string argument*/
void read_json_file(const char* const, JsonValue&, JsonAllocator&, std::string&,
                    int root_type);

bool try_read_float(JsonNode&, float&, const char*);
// (unsigned int)node->value.toNumber();
bool try_read_uint(JsonNode&, unsigned int&, const char*);
bool try_read_vector(JsonNode&, std::vector<float>&, const char*);
bool try_read_string(JsonNode&, std::string&, const char*);

///
/// Cmd line args parsing
///
struct ArgOption {
  bool _required = false;
  std::string _name = "";
  std::string _help = "";
  std::vector<std::string> _mnemonics;

  ArgOption& help(const char*);
  ArgOption& required();
};

class Argparse {
  typedef std::pair<ArgOption*, std::string> ArgValue;

 public:
  Argparse(const char*, const char*);

  ArgOption& add_argument(const char*);
  ArgOption& add_argument(const char*, const char*);
  bool parse(size_t, char**);
  void print_help();

  bool has_arg(const char*);
  const char* value(const char*);
  void value(const char*, size_t&);

 private:
  ArgOption& add_argument(size_t, const char**);
  ArgValue* get_value(const char*);

  const std::string _general_help, _exec_name;
  std::vector<ArgOption> _options;
  std::vector<ArgValue> _values;
};
}
}

#endif /* PCH_H   */
