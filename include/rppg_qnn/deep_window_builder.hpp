#pragma once

#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <deque>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

namespace rppg_qnn {

class DeepWindowBuilder {
 public:
  DeepWindowBuilder(double window_sec, std::size_t sample_count,
                    cv::Size output_size);

  [[nodiscard]] std::optional<DeepInput> add_roi(const RoiPacket& packet);
  [[nodiscard]] std::string status() const;

 private:
  struct Frame {
    double timestamp_sec;
    cv::Mat bgr;
  };

  double window_sec_;
  std::size_t sample_count_;
  cv::Size output_size_;
  std::deque<Frame> frames_;
  std::string status_{"sampling"};
};

}  // namespace rppg_qnn
