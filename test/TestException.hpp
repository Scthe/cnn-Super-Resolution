#ifndef TEST_EXCEPTION_H
#define TEST_EXCEPTION_H

#include <stdexcept>
#include <sstream>

namespace test {
class TestException : public std::runtime_error {
 public:
  TestException();
  TestException(const char*);
  TestException(const TestException&);

  virtual const char* what() const throw();

 private:
  std::ostringstream cnvt;
};
}
#endif /* TEST_EXCEPTION_H   */
