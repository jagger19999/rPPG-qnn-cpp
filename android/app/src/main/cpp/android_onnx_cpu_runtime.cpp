#include "android_onnx_cpu_runtime.hpp"

#include "rppg_qnn/error.hpp"
#include "rppg_qnn/tscan_runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn::android {
namespace {

constexpr std::size_t kFrames = 180U;
constexpr std::size_t kChannels = 6U;
constexpr std::size_t kHeight = 72U;
constexpr std::size_t kWidth = 72U;
constexpr std::size_t kInputValues = kFrames * kChannels * kHeight * kWidth;

class OnnxCpuSession final : public ITscanSession {
 public:
  explicit OnnxCpuSession(const std::string& model_path)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "rppg-tscan"),
        session_options_(),
        session_(nullptr) {
    if (!std::filesystem::is_regular_file(model_path)) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     "TSCAN ONNX model is missing: " + model_path);
    }
    session_options_.SetIntraOpNumThreads(2);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(environment_, model_path.c_str(), session_options_);
  }

  [[nodiscard]] std::string backend_name() const override {
    return "ONNX_RUNTIME_CPU";
  }

  TscanModelOutput run(const std::vector<float>& values,
                       const std::vector<std::int64_t>& shape) override {
    if (shape != std::vector<std::int64_t>{180, 6, 72, 72} ||
        values.size() != kInputValues) {
      throw AppError(ErrorCode::InferenceFailed,
                     "TSCAN input must be float32 [180,6,72,72]");
    }
    try {
      Ort::MemoryInfo memory =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory, const_cast<float*>(values.data()), values.size(), shape.data(),
          shape.size());
      constexpr const char* input_names[] = {"frames"};
      constexpr const char* output_names[] = {"pulse"};
      std::vector<Ort::Value> outputs =
          session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1U,
                       output_names, 1U);
      if (outputs.size() != 1U || !outputs.front().IsTensor()) {
        throw AppError(ErrorCode::InferenceFailed,
                       "TSCAN did not return one pulse tensor");
      }
      const Ort::TensorTypeAndShapeInfo info =
          outputs.front().GetTensorTypeAndShapeInfo();
      const std::vector<std::int64_t> output_shape = info.GetShape();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          output_shape != std::vector<std::int64_t>{180, 1}) {
        throw AppError(ErrorCode::InferenceFailed,
                       "TSCAN output must be float32 [180,1]");
      }
      const float* output = outputs.front().GetTensorData<float>();
      std::vector<float> waveform(output, output + kFrames);
      if (!std::all_of(waveform.begin(), waveform.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw AppError(ErrorCode::InferenceFailed,
                       "TSCAN output contains non-finite values");
      }
      return {std::move(waveform), output_shape};
    } catch (const AppError&) {
      throw;
    } catch (const Ort::Exception& error) {
      throw AppError(ErrorCode::InferenceFailed,
                     std::string("ONNX Runtime CPU failed: ") + error.what());
    }
  }

 private:
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    const std::string& model_path) {
  return make_tscan_runtime(std::make_unique<OnnxCpuSession>(model_path));
}

}  // namespace rppg_qnn::android
