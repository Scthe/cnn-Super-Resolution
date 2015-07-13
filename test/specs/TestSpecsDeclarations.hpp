#ifndef TEST_SPECS_DECL_H
#define TEST_SPECS_DECL_H

#include "../TestRunner.hpp"
#include "../../src/opencl/Context.hpp"
#include <string>

#define DECLARE_TEST_SPEC(X, ...)                                              \
  struct CONCATENATE(X, Impl);                                                 \
  class X : public TestCase {                                                  \
   public:                                                                     \
    X();                                                                       \
    ~X();                                                                      \
    void init(__VA_ARGS__);                                                    \
    std::string name(size_t data_set_id) override;                             \
    bool operator()(size_t data_set_id, cnn_sr::DataPipeline *const) override; \
    size_t data_set_count() override;                                          \
                                                                               \
   private:                                                                    \
    CONCATENATE(X, Impl) *const _impl = nullptr;                               \
  };

#define TEST_SPEC_PIMPL(X)                      \
  X::X() : _impl(new CONCATENATE(X, Impl)()) {} \
  X::~X() { delete _impl; }

namespace test {
namespace specs {

DECLARE_TEST_SPEC(ExtractLumaTest)
DECLARE_TEST_SPEC(MeanSquaredErrorTest)
DECLARE_TEST_SPEC(SubtractFromAllTest)
DECLARE_TEST_SPEC(SumTest)
DECLARE_TEST_SPEC(LayerDeltasTest)
DECLARE_TEST_SPEC(BackpropagationTest)
DECLARE_TEST_SPEC(LayerTest)
DECLARE_TEST_SPEC(LastLayerDeltaTest)
DECLARE_TEST_SPEC(WeightDecayTest)
DECLARE_TEST_SPEC(UpdateParametersTest)
DECLARE_TEST_SPEC(ConfigTest)

}
}

#endif /* TEST_SPECS_DECL_H   */
