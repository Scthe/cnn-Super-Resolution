#ifndef __UTILS__H
#define __UTILS__H

#include <sstream>
#include <cstring>
#include <string>
#include <vector>

// TODO use dropbox/json11 since it allows for write too
union JsonValue;
class JsonAllocator;

namespace cnn_sr {
namespace utils {

void get_file_content(const char* const, std::stringstream&);

void read_json_file(const char* const, JsonValue&, JsonAllocator&,
                    std::string& file_content, int root_type);

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
}
}

// it's easier then including header file just for one typedef,
// since we can't typedef member class
#define IOException std::ios_base::failure

#define NUM_ELEMS(x) (sizeof(x) / sizeof((x)[0]))

// TODO change to: try_read_as_float(JsonNode& float&, string json_name)
#define JSON_READ_FLOAT(NODE, LHS, PROP_NAME) \
  if (strcmp(NODE->key, PROP_NAME) == 0 &&    \
      NODE->value.getTag() == JSON_NUMBER) {  \
    LHS = (float)NODE->value.toNumber();      \
  }

#define JSON_READ_UINT(NODE, OBJECT, PROP_NAME)              \
  if (strcmp(NODE->key, #PROP_NAME) == 0 &&                  \
      NODE->value.getTag() == JSON_NUMBER) {                 \
    OBJECT.PROP_NAME = (unsigned int)NODE->value.toNumber(); \
  }

#define JSON_READ_NUM_ARRAY(NODE, OBJECT, PROP_NAME)   \
  if (strcmp(NODE->key, #PROP_NAME) == 0 &&            \
      NODE->value.getTag() == JSON_ARRAY) {            \
    auto arr_raw = NODE->value;                        \
    for (auto val : arr_raw) {                         \
      /* ASSERT(val->value.getTag() == JSON_NUMBER);*/ \
      double num = val->value.toNumber();              \
      OBJECT.PROP_NAME.push_back(num);                 \
    }                                                  \
  }

#define JSON_READ_STR(NODE, OBJECT, PROP_NAME) \
  if (strcmp(NODE->key, #PROP_NAME) == 0 &&    \
      NODE->value.getTag() == JSON_STRING) {   \
    OBJECT.PROP_NAME = NODE->value.toString(); \
  }

#endif /* __UTILS__H   */
