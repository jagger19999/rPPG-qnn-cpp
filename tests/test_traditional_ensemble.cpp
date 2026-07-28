#include "rppg_qnn/traditional_ensemble.hpp"

#include <cmath>
#include <cstddef>

#include <opencv2/core.hpp>

#include "test_support.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main() {
  rppg_qnn::TraditionalEnsemble ensemble;
  for (std::size_t index = 0; index <= 600U; ++index) {
    const double time = static_cast<double>(index) / 30.0;
    const double pulse = std::sin(2.0 * kPi * 1.2 * time);
    const double motion = std::sin(2.0 * kPi * 0.2 * time);
    const cv::Scalar bgr(70.0 * (1.0 + 0.04 * motion - 0.003 * pulse),
                         90.0 * (1.0 + 0.04 * motion + 0.010 * pulse),
                         120.0 * (1.0 + 0.04 * motion + 0.002 * pulse));
    rppg_qnn::FrameQualitySample quality;
    quality.face_area_ratio = 0.2;
    quality.motion_px = 2.0;
    quality.face_count = 1;
    ensemble.add_sample(time, bgr, quality);
  }
  const auto snapshot = ensemble.latest_snapshot();
  EXPECT_TRUE(snapshot.has_value());
  if (snapshot.has_value()) {
    EXPECT_EQ(snapshot->candidates.size(), std::size_t{4});
    EXPECT_TRUE(snapshot->gate.accepted);
    EXPECT_TRUE(snapshot->accepted_available);
    EXPECT_TRUE(std::abs(snapshot->accepted_bpm - 72.0) < 1.0);
  }
  return test_support::finish();
}
