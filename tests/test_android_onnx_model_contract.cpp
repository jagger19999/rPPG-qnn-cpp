#include "android_onnx_model_contract.hpp"

#include "rppg_qnn/error.hpp"
#include "test_support.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using rppg_qnn::DeepModel;
using rppg_qnn::ErrorCode;
using rppg_qnn::android::OnnxTensorContract;
using rppg_qnn::android::onnx_model_contract;
using rppg_qnn::android::validate_onnx_model_contract;

void contracts_are_exact_and_model_specific() {
  const auto& tscan = onnx_model_contract(DeepModel::Tscan);
  EXPECT_EQ(tscan.input.name, std::string("frames"));
  EXPECT_EQ(tscan.input.element_type, OnnxTensorContract::ElementType::Float32);
  EXPECT_EQ(tscan.input.shape, (std::vector<std::int64_t>{180, 6, 72, 72}));
  EXPECT_EQ(tscan.output.name, std::string("pulse"));
  EXPECT_EQ(tscan.output.element_type, OnnxTensorContract::ElementType::Float32);
  EXPECT_EQ(tscan.output.shape, (std::vector<std::int64_t>{180, 1}));

  const auto& efficientphys = onnx_model_contract(DeepModel::EfficientPhys);
  EXPECT_EQ(efficientphys.input.name, std::string("frames"));
  EXPECT_EQ(efficientphys.input.element_type,
            OnnxTensorContract::ElementType::Float32);
  EXPECT_EQ(efficientphys.input.shape,
            (std::vector<std::int64_t>{181, 3, 72, 72}));
  EXPECT_EQ(efficientphys.output.name, std::string("pulse"));
  EXPECT_EQ(efficientphys.output.element_type,
            OnnxTensorContract::ElementType::Float32);
  EXPECT_EQ(efficientphys.output.shape,
            (std::vector<std::int64_t>{180, 1}));
}

void mismatched_metadata_is_rejected() {
  auto actual = onnx_model_contract(DeepModel::Tscan);
  actual.input.shape = {181, 3, 72, 72};
  EXPECT_APP_ERROR(validate_onnx_model_contract(DeepModel::Tscan, actual),
                   ErrorCode::ModelLoadFailed);

  actual = onnx_model_contract(DeepModel::EfficientPhys);
  actual.output.name = "wrong";
  EXPECT_APP_ERROR(
      validate_onnx_model_contract(DeepModel::EfficientPhys, actual),
      ErrorCode::ModelLoadFailed);

  actual = onnx_model_contract(DeepModel::Tscan);
  actual.input.element_type =
      static_cast<OnnxTensorContract::ElementType>(99);
  EXPECT_APP_ERROR(validate_onnx_model_contract(DeepModel::Tscan, actual),
                   ErrorCode::ModelLoadFailed);
}

void disabled_has_no_onnx_contract() {
  EXPECT_APP_ERROR(onnx_model_contract(DeepModel::Disabled),
                   ErrorCode::ConfigInvalid);
}

}  // namespace

int main() {
  contracts_are_exact_and_model_specific();
  mismatched_metadata_is_rejected();
  disabled_has_no_onnx_contract();
  return test_support::finish();
}
