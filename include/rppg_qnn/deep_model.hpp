#pragma once

#include <string_view>

namespace rppg_qnn {

enum class DeepModel {
  Disabled,
  Tscan,
  EfficientPhys,
};

DeepModel parse_deep_model(std::string_view value);
std::string_view to_string(DeepModel model) noexcept;

}  // namespace rppg_qnn
