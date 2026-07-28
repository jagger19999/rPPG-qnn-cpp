#include "android_onnx_cpu_runtime.hpp"

#include "android_onnx_model_contract.hpp"

#include "rppg_qnn/efficientphys_runtime.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/tscan_runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
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

#ifndef RPPG_TSCAN_ORT_INTRA_OP_THREADS
#define RPPG_TSCAN_ORT_INTRA_OP_THREADS 2
#endif

#ifndef RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS
#define RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS 2
#endif

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1},
      [](std::size_t product, std::int64_t dimension) {
        return product * static_cast<std::size_t>(dimension);
      });
}

struct OnnxRunOutput {
  std::vector<float> values;
  double runtime_ms{0.0};
};

class OnnxCpuSession final {
 public:
  OnnxCpuSession(DeepModel model, const std::string& model_path,
                 OrtThreadOptions thread_options) try
      : model_(model),
        contract_(onnx_model_contract(model)),
        environment_(ORT_LOGGING_LEVEL_WARNING, "rppg-deep"),
        session_options_(),
        session_(nullptr) {
    if (!valid_ort_thread_options(thread_options)) {
      throw AppError(
          ErrorCode::ConfigInvalid,
          "ORT threads require intra-op 2, 4, or 6 and inter-op exactly 1");
    }
    if (!std::filesystem::is_regular_file(model_path)) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(model)) +
                         " ONNX model is missing: " + model_path);
    }
    session_options_.SetIntraOpNumThreads(thread_options.intra_op_threads);
    session_options_.SetInterOpNumThreads(thread_options.inter_op_threads);
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

  OnnxRunOutput run(const std::vector<float>& values,
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
      const auto runtime_started = std::chrono::steady_clock::now();
      std::vector<Ort::Value> outputs = session_.Run(
          Ort::RunOptions{nullptr}, input_names, &input_tensor, 1U, output_names,
          1U);
      const double runtime_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - runtime_started)
              .count();
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
      return {std::move(waveform), runtime_ms};
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
  TscanOnnxSession(const std::string& model_path, OrtThreadOptions options)
      : session_(DeepModel::Tscan, model_path, options) {}

  [[nodiscard]] std::string backend_name() const override {
    return "ONNX_RUNTIME_CPU";
  }

  TscanModelOutput run(const std::vector<float>& values,
                       const std::vector<std::int64_t>& shape) override {
    OnnxRunOutput output = session_.run(values, shape);
    return {std::move(output.values),
            onnx_model_contract(DeepModel::Tscan).output.shape,
            output.runtime_ms};
  }

 private:
  OnnxCpuSession session_;
};

class EfficientPhysOnnxSession final : public IEfficientPhysSession {
 public:
  EfficientPhysOnnxSession(const std::string& model_path,
                           OrtThreadOptions options)
      : session_(DeepModel::EfficientPhys, model_path, options) {}

  [[nodiscard]] std::string backend_name() const override {
    return "onnxruntime_cpu";
  }

  EfficientPhysModelOutput run(
      const std::vector<float>& values,
      const std::vector<std::int64_t>& shape) override {
    OnnxRunOutput output = session_.run(values, shape);
    return {std::move(output.values),
            onnx_model_contract(DeepModel::EfficientPhys).output.shape,
            output.runtime_ms};
  }

 private:
  OnnxCpuSession session_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    DeepModel model, const std::string& model_path) {
  const int model_threads =
      model == DeepModel::EfficientPhys
          ? RPPG_EFFICIENTPHYS_ORT_INTRA_OP_THREADS
          : RPPG_TSCAN_ORT_INTRA_OP_THREADS;
  return make_onnx_cpu_runtime(model, model_path, {model_threads, 1});
}

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    DeepModel model, const std::string& model_path, OrtThreadOptions options) {
  switch (model) {
    case DeepModel::Tscan:
      return make_tscan_runtime(
          std::make_unique<TscanOnnxSession>(model_path, options));
    case DeepModel::EfficientPhys:
      return make_onnxruntime_efficientphys_runtime(
          std::make_unique<EfficientPhysOnnxSession>(model_path, options));
    case DeepModel::Disabled:
      throw AppError(ErrorCode::ConfigInvalid,
                     "disabled deep model cannot create an ONNX runtime");
  }
  throw AppError(ErrorCode::ConfigInvalid,
                 "unknown deep model cannot create an ONNX runtime");
}

}  // namespace rppg_qnn::android
