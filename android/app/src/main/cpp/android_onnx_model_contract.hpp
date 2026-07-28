#pragma once

#include "rppg_qnn/deep_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rppg_qnn::android {

struct OnnxTensorContract {
  enum class ElementType { Float32 };

  std::string name;
  ElementType element_type{ElementType::Float32};
  std::vector<std::int64_t> shape;

  friend bool operator==(const OnnxTensorContract& lhs,
                         const OnnxTensorContract& rhs) {
    return lhs.name == rhs.name && lhs.element_type == rhs.element_type &&
           lhs.shape == rhs.shape;
  }
};

struct OnnxModelContract {
  OnnxTensorContract input;
  OnnxTensorContract output;
};

const OnnxModelContract& onnx_model_contract(DeepModel model);
void validate_onnx_model_contract(DeepModel selected_model,
                                  const OnnxModelContract& actual);

}  // namespace rppg_qnn::android
