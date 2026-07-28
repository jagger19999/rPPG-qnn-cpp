#pragma once

#include "rppg_qnn/contracts.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace rppg_qnn {

struct DeepInput {
  double start_sec{0.0};
  double end_sec{0.0};
  double source_fps{0.0};
  double max_frame_gap_sec{0.0};
  std::size_t source_frame_count{0};
  double window_materialization_ms{0.0};
  std::vector<float> tensor;
  std::vector<std::int64_t> shape;
};

class IDeepRuntime {
 public:
  virtual ~IDeepRuntime() = default;
  [[nodiscard]] virtual std::string backend_name() const = 0;
  virtual HeartRateResult infer(const DeepInput& input) = 0;
};

std::unique_ptr<IDeepRuntime> make_fake_deep_runtime(
    std::chrono::milliseconds latency);

}  // namespace rppg_qnn
