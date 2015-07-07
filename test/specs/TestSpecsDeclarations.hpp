#ifndef TEST_SPECS_DECL_H
#define TEST_SPECS_DECL_H

#include "../../src/opencl/Context.hpp"
#include "../TestRunner.hpp"

namespace cnn_sr {
class DataPipeline;
}

#define DECLARE_TEST_SPEC(X, ...)                      \
  struct CONCATENATE(X, Impl);                         \
  struct X : TestCase {                                \
    X();                                               \
    ~X();                                              \
    void init(__VA_ARGS__);                            \
    const char *name() override;                       \
    bool operator()(opencl::Context * const) override; \
                                                       \
   private:                                            \
    CONCATENATE(X, Impl) *const _impl = nullptr;       \
  };

#define TEST_SPEC_PIMPL(X)                      \
  X::X() : _impl(new CONCATENATE(X, Impl)()) {} \
  X::~X() { delete _impl; }

namespace test {
namespace specs {

DECLARE_TEST_SPEC(LayerDeltasTest, cnn_sr::DataPipeline *)
DECLARE_TEST_SPEC(BackpropagationTest, cnn_sr::DataPipeline *)
}
}

#endif /* TEST_SPECS_DECL_H   */
