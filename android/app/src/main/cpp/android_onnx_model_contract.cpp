#include "android_onnx_model_contract.hpp"

#include "rppg_qnn/error.hpp"

#include <string>

namespace rppg_qnn::android {
namespace {

const OnnxModelContract kTscanContract{
    {"frames", OnnxTensorContract::ElementType::Float32, {180, 6, 72, 72}},
    {"pulse", OnnxTensorContract::ElementType::Float32, {180, 1}}};
const OnnxModelContract kEfficientPhysContract{
    {"frames", OnnxTensorContract::ElementType::Float32, {181, 3, 72, 72}},
    {"pulse", OnnxTensorContract::ElementType::Float32, {180, 1}}};

}  // namespace

const OnnxModelContract& onnx_model_contract(DeepModel model) {
  switch (model) {
    case DeepModel::Tscan:
      return kTscanContract;
    case DeepModel::EfficientPhys:
      return kEfficientPhysContract;
    case DeepModel::Disabled:
      throw AppError(ErrorCode::ConfigInvalid,
                     "disabled deep model has no ONNX contract");
  }
  throw AppError(ErrorCode::ConfigInvalid, "unknown deep model contract");
}

void validate_onnx_model_contract(DeepModel selected_model,
                                  const OnnxModelContract& actual) {
  const OnnxModelContract& expected = onnx_model_contract(selected_model);
  if (!(actual.input == expected.input) || !(actual.output == expected.output)) {
    throw AppError(
        ErrorCode::ModelLoadFailed,
        "selected " + std::string(to_string(selected_model)) +
            " model has incompatible ONNX input/output name, type, or shape");
  }
}

}  // namespace rppg_qnn::android
