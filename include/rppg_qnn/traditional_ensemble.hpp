#pragma once

#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/measurement_engine.hpp"
#include "rppg_qnn/traditional_predictor.hpp"

#include <cstddef>
#include <deque>
#include <optional>

#include <opencv2/core.hpp>

namespace rppg_qnn {

struct FrameQualitySample {
  double face_area_ratio{0.0};
  double motion_px{0.0};
  int face_count{0};
};

class TraditionalEnsemble {
 public:
  explicit TraditionalEnsemble(QualityProfile profile = {});

  void add_sample(double timestamp_sec, const cv::Scalar& mean_bgr,
                  FrameQualitySample quality);
  [[nodiscard]] std::optional<MeasurementSnapshot> latest_snapshot() const;
  [[nodiscard]] std::size_t evaluation_count() const;
  void reset();

 private:
  struct Sample {
    double timestamp_sec{0.0};
    cv::Vec3d rgb;
    FrameQualitySample quality;
  };

  void evaluate(double timestamp_sec);

  TraditionalPredictor green_{TraditionalMethod::Green};
  TraditionalPredictor pos_{TraditionalMethod::Pos};
  TraditionalPredictor chrom_{TraditionalMethod::Chrom};
  TraditionalPredictor lgi_{TraditionalMethod::Lgi};
  MeasurementEngine measurement_;
  std::deque<Sample> samples_;
  std::optional<MeasurementSnapshot> latest_;
  std::size_t evaluation_count_{0};
  std::size_t predictor_evaluation_count_{0};
};

}  // namespace rppg_qnn
