#pragma once

#include "rppg_qnn/deep_runtime.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rppg_qnn {

struct TscanModelOutput {
  std::vector<float> waveform;
  std::vector<std::int64_t> shape;
};

class ITscanSession {
 public:
  virtual ~ITscanSession() = default;
  [[nodiscard]] virtual std::string backend_name() const = 0;
  virtual TscanModelOutput run(const std::vector<float>& values,
                               const std::vector<std::int64_t>& shape) = 0;
};

std::unique_ptr<IDeepRuntime> make_tscan_runtime(
    std::unique_ptr<ITscanSession> session);

}  // namespace rppg_qnn
