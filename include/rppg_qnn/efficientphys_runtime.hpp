#pragma once

#include "rppg_qnn/deep_runtime.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rppg_qnn {

struct EfficientPhysModelOutput {
  std::vector<float> waveform;
  std::vector<std::int64_t> shape;
};

class IEfficientPhysSession {
 public:
  virtual ~IEfficientPhysSession() = default;
  [[nodiscard]] virtual std::string backend_name() const = 0;
  virtual EfficientPhysModelOutput run(
      const std::vector<float>& values,
      const std::vector<std::int64_t>& shape) = 0;
};

std::unique_ptr<IDeepRuntime> make_onnxruntime_efficientphys_runtime(
    std::unique_ptr<IEfficientPhysSession> session);

}  // namespace rppg_qnn
