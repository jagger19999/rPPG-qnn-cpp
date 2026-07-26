#include "android_onnx_cpu_runtime.hpp"

#include "android_onnx_model_contract.hpp"

#include "rppg_qnn/efficientphys_runtime.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/tscan_runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn::android {
namespace {

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1},
      [](std::size_t product, std::int64_t dimension) {
        return product * static_cast<std::size_t>(dimension);
      });
}

class OnnxCpuSession final {
 public:
  OnnxCpuSession(DeepModel model, const std::string& model_path) try
      : model_(model),
        contract_(onnx_model_contract(model)),
        environment_(ORT_LOGGING_LEVEL_WARNING, "rppg-deep"),
        session_options_(),
        session_(nullptr) {
    if (!std::filesystem::is_regular_file(model_path)) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(model)) +
                         " ONNX model is missing: " + model_path);
    }
    session_options_.SetIntraOpNumThreads(2);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(environment_, model_path.c_str(), session_options_);
    validate_loaded_model();
  } catch (const AppError&) {
    throw;
  } catch (const Ort::Exception& error) {
    throw AppError(ErrorCode::ModelLoadFailed,
                   std::string("ONNX Runtime CPU model load failed: ") +
                       error.what());
  }

  std::vector<float> run(const std::vector<float>& values,
                         const std::vector<std::int64_t>& shape) {
    if (shape != contract_.input.shape ||
        values.size() != element_count(contract_.input.shape)) {
      throw AppError(ErrorCode::InferenceFailed,
                     std::string(to_string(model_)) +
                         " input shape/value count violates its ONNX contract");
    }
    try {
      Ort::MemoryInfo memory =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory, const_cast<float*>(values.data()), values.size(), shape.data(),
          shape.size());
      const char* input_names[] = {contract_.input.name.c_str()};
      const char* output_names[] = {contract_.output.name.c_str()};
      std::vector<Ort::Value> outputs = session_.Run(
          Ort::RunOptions{nullptr}, input_names, &input_tensor, 1U, output_names,
          1U);
      if (outputs.size() != 1U || !outputs.front().IsTensor()) {
        throw AppError(ErrorCode::InferenceFailed,
                       std::string(to_string(model_)) +
                           " did not return one pulse tensor");
      }
      const Ort::TensorTypeAndShapeInfo info =
          outputs.front().GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          info.GetShape() != contract_.output.shape) {
        throw AppError(ErrorCode::InferenceFailed,
                       std::string(to_string(model_)) +
                           " output type/shape violates its ONNX contract");
      }
      const std::size_t count = element_count(contract_.output.shape);
      const float* output = outputs.front().GetTensorData<float>();
      std::vector<float> waveform(output, output + count);
      if (!std::all_of(waveform.begin(), waveform.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw AppError(ErrorCode::InferenceFailed,
                       std::string(to_string(model_)) +
                           " output contains non-finite values");
      }
      return waveform;
    } catch (const AppError&) {
      throw;
    } catch (const Ort::Exception& error) {
      throw AppError(ErrorCode::InferenceFailed,
                     std::string("ONNX Runtime CPU failed: ") + error.what());
    }
  }

 private:
  OnnxTensorContract read_tensor_contract(bool input) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count =
        input ? session_.GetInputCount() : session_.GetOutputCount();
    if (count != 1U) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(model_)) +
                         " ONNX model must have exactly one input and output");
    }
    Ort::AllocatedStringPtr name =
        input ? session_.GetInputNameAllocated(0U, allocator)
              : session_.GetOutputNameAllocated(0U, allocator);
    Ort::TypeInfo type =
        input ? session_.GetInputTypeInfo(0U) : session_.GetOutputTypeInfo(0U);
    const auto tensor = type.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(model_)) +
                         " ONNX input/output type must be float32");
    }
    return {name.get(), OnnxTensorContract::ElementType::Float32,
            tensor.GetShape()};
  }

  void validate_loaded_model() {
    if (session_.GetInputCount() != 1U || session_.GetOutputCount() != 1U) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(model_)) +
                         " ONNX model must have exactly one input and output");
    }
    validate_onnx_model_contract(
        model_, {read_tensor_contract(true), read_tensor_contract(false)});
  }

  DeepModel model_;
  OnnxModelContract contract_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
};

class TscanOnnxSession final : public ITscanSession {
 public:
  explicit TscanOnnxSession(const std::string& model_path)
      : session_(DeepModel::Tscan, model_path) {}

  [[nodiscard]] std::string backend_name() const override {
    return "ONNX_RUNTIME_CPU";
  }

  TscanModelOutput run(const std::vector<float>& values,
                       const std::vector<std::int64_t>& shape) override {
    return {session_.run(values, shape),
            onnx_model_contract(DeepModel::Tscan).output.shape};
  }

 private:
  OnnxCpuSession session_;
};

class EfficientPhysOnnxSession final : public IEfficientPhysSession {
 public:
  explicit EfficientPhysOnnxSession(const std::string& model_path)
      : session_(DeepModel::EfficientPhys, model_path) {}

  [[nodiscard]] std::string backend_name() const override {
    return "onnxruntime_cpu";
  }

  EfficientPhysModelOutput run(
      const std::vector<float>& values,
      const std::vector<std::int64_t>& shape) override {
    return {session_.run(values, shape),
            onnx_model_contract(DeepModel::EfficientPhys).output.shape};
  }

 private:
  OnnxCpuSession session_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    DeepModel model, const std::string& model_path) {
  switch (model) {
    case DeepModel::Tscan:
      return make_tscan_runtime(std::make_unique<TscanOnnxSession>(model_path));
    case DeepModel::EfficientPhys:
      return make_onnxruntime_efficientphys_runtime(
          std::make_unique<EfficientPhysOnnxSession>(model_path));
    case DeepModel::Disabled:
      throw AppError(ErrorCode::ConfigInvalid,
                     "disabled deep model cannot create an ONNX runtime");
  }
  throw AppError(ErrorCode::ConfigInvalid,
                 "unknown deep model cannot create an ONNX runtime");
}

}  // namespace rppg_qnn::android
