#include "rppg_qnn/deep_model.hpp"
#include "rppg_qnn/error.hpp"

#include "test_support.hpp"

#include <array>
#include <string_view>

namespace {

using rppg_qnn::DeepModel;
using rppg_qnn::ErrorCode;
using rppg_qnn::parse_deep_model;
using rppg_qnn::to_string;

void canonical_values_parse_and_stringify() {
  constexpr std::array<std::pair<std::string_view, DeepModel>, 3> cases{{
      {"disabled", DeepModel::Disabled},
      {"tscan", DeepModel::Tscan},
      {"efficientphys", DeepModel::EfficientPhys},
  }};

  for (const auto& [text, model] : cases) {
    EXPECT_EQ(parse_deep_model(text), model);
    EXPECT_EQ(to_string(model), text);
  }
}

void noncanonical_values_are_rejected() {
  constexpr std::array<std::string_view, 6> invalid_values{{
      "", "TSCAN", "efficient_phys", "onnxruntime_cpu", "none", "tscan ",
  }};

  for (const std::string_view value : invalid_values) {
    EXPECT_APP_ERROR(parse_deep_model(value), ErrorCode::ConfigInvalid);
  }
}

void invalid_enum_values_are_rejected() {
  EXPECT_APP_ERROR(to_string(static_cast<DeepModel>(99)),
                   ErrorCode::ConfigInvalid);
}

}  // namespace

int main() {
  canonical_values_parse_and_stringify();
  noncanonical_values_are_rejected();
  invalid_enum_values_are_rejected();
  return test_support::finish();
}
