#include "TestSpecsDeclarations.hpp"
#include "../../src/Config.hpp"

namespace test {
namespace specs {

///
/// Data set
///
struct ConfigDataSet : DataSet {
  ConfigDataSet(std::string name, const char* cfg_file, bool expect_io_error,
                bool expect_invalid_val)
      : DataSet(name),
        cfg_file(cfg_file),
        expect_io_error(expect_io_error),
        expect_invalid_val(expect_invalid_val) {}

  const char* cfg_file;
  bool expect_io_error, expect_invalid_val;
};

///
/// PIMPL
///
struct ConfigTestImpl {
  /* clang-format off */
  ConfigDataSet data_sets[4] = {
      ConfigDataSet("ok", "test/data/config.json", false,false),
      ConfigDataSet("invalid value", "test/data/config_invalid_val.json", false,true),
      ConfigDataSet("invalid file", "test/data/config_non_parseable.json", true,false),
      ConfigDataSet("file nonexistent", "test/data/NOPE.json", true,false)};
  /* clang-format on */

  cnn_sr::ParametersDistribution pd1 = {0.9, 0.9, 0.9, 0.9};
  cnn_sr::ParametersDistribution pd2 = {2.001, 2.001, 2.001, 2.001};
  cnn_sr::ParametersDistribution pd3 = {0.001, 0.001, 0.001, 0.001};
  float learning_rates[3] = {12, 34, 56};
  /* clang-format off */
  cnn_sr::Config correct_result={32, 16,
                                 9, 1, 5,
                                 123.5f,0.1f, learning_rates,
                                 pd1, pd2, pd3,
                                 "cnn-parameters-a.json"};
  /* clang-format on */
};

///
/// ConfigTest
///

TEST_SPEC_PIMPL(ConfigTest)

void ConfigTest::init() {}

size_t ConfigTest::data_set_count() { return 4; }

std::string ConfigTest::name(size_t data_set_id) {
  assert_data_set_ok(data_set_id);
  return "Config test - " + _impl->data_sets[data_set_id].name;
}

bool params_cmp(cnn_sr::ParametersDistribution a,
                cnn_sr::ParametersDistribution b) {
  return a.mean_w == b.mean_w && a.sd_w == b.sd_w &&  //
         a.mean_b == b.mean_b && a.sd_b == b.sd_b;
}

bool ConfigTest::operator()(size_t data_set_id,
                            cnn_sr::DataPipeline* const pipeline) {
  using namespace cnn_sr;
  assert_not_null(pipeline);
  assert_data_set_ok(data_set_id);
  auto data = _impl->data_sets[data_set_id];
  Config& c2 = _impl->correct_result;

  bool io_err = false, invalid_val = false;
  ConfigReader reader;

  try {
    Config c1 = reader.read(data.cfg_file);
    assert_true(c1.n1 == c2.n1 && c1.n2 == c2.n2,
                "filter count does not match");
    assert_true(c1.f1 == c2.f1 && c1.f2 == c2.f2 && c1.f3 == c2.f3,
                "filter spatial size does not match");
    assert_true(c1.momentum == c2.momentum, "momentum does not match");
    assert_true(c1.weight_decay_parameter == c2.weight_decay_parameter,
                "weight decay parameter does not match");
    assert_true(c1.learning_rate[0] == c2.learning_rate[0]         //
                    && c1.learning_rate[1] == c2.learning_rate[1]  //
                    && c1.learning_rate[2] == c2.learning_rate[2],
                "learning rate does not match");
    // std::cout << c1.parameters_file << "'" << std::endl;
    // std::cout << c2.parameters_file << "'" << std::endl;
    assert_true(c1.parameters_file.compare(c2.parameters_file) == 0,
                "parameters_file does not match");
    assert_true(params_cmp(c1.params_distr_1, c2.params_distr_1),
                "parameters distribution 1 does not match");
    assert_true(params_cmp(c1.params_distr_2, c2.params_distr_2),
                "parameters distribution 2 does not match");
    assert_true(params_cmp(c1.params_distr_3, c2.params_distr_3),
                "parameters distribution 3 does not match");
  } catch (TestException& e) {
    std::cout << e.what() << std::endl;
    invalid_val = true;
  } catch (IOException& e) {
    std::cout << e.what() << std::endl;
    io_err = true;
  } /* catch (...) {
     assert_true(false, "Unknown error");
   }*/

  assert_true(io_err == data.expect_io_error, "Expected IO error");
  assert_true(invalid_val == data.expect_invalid_val,
              "Expected values mismatch");
  return true;
}

//
//
}  // namespace specs
}  // namespace test
