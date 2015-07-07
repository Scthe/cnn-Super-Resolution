#ifndef TEST_EXCEPTION_H
#define TEST_EXCEPTION_H

#include <stdexcept>
#include <sstream>

template <typename T>
class TestException : public std::runtime_error {
 public:
  TestException(T x, T y)                //
      : runtime_error("TestException"),  //
        _x(x),
        _y(y) {}

  TestException(T x, T y, const char* msg)  //
      : runtime_error("TestException"),     //
        _x(x),
        _y(y) {
    cnvt.str("");
    cnvt << runtime_error::what() << ": " << msg;
  }

  TestException(const char* msg)  //
      : runtime_error("TestException") {
    cnvt.str("");
    cnvt << runtime_error::what() << ": " << msg;
  }

  virtual const char* what() const throw() {
    // cnvt << runtime_error::what() << ": ";
    return cnvt.str().c_str();
  }

 private:
  T _x;
  T _y;

  static std::ostringstream cnvt;
};

template <typename T>
std::ostringstream TestException<T>::cnvt;

#endif /* TEST_EXCEPTION_H   */
