#include "rppg_qnn/measurement_engine.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "test_support.hpp"

namespace {

rppg_qnn::CandidateResult candidate(const std::string& method, double bpm,
                                    double confidence = 0.8) {
  rppg_qnn::CandidateResult result;
  result.method = method;
  result.bpm = bpm;
  result.confidence = confidence;
  result.peak_ratio = confidence;
  result.valid = true;
  return result;
}

rppg_qnn::QualityMetrics good_quality() {
  rppg_qnn::QualityMetrics quality;
  quality.brightness = 0.5;
  quality.signal_std = 0.02;
  quality.spectral_peak_ratio = 0.4;
  quality.face_area_ratio = 0.2;
  quality.motion_px = 2.0;
  quality.source_fps = 30.0;
  quality.max_frame_gap_sec = 1.0 / 30.0;
  quality.face_count = 1;
  return quality;
}

void accepts_consensus_and_exposes_shadow_router() {
  rppg_qnn::MeasurementEngine engine;
  const auto snapshot = engine.evaluate(
      {candidate("GREEN", 73.0), candidate("POS", 72.0),
       candidate("CHROM", 74.0), candidate("LGI", 71.0)},
      good_quality(), 10.0);
  EXPECT_TRUE(snapshot.gate.accepted);
  EXPECT_TRUE(snapshot.accepted_available);
  EXPECT_TRUE(snapshot.display_available);
  EXPECT_TRUE(!snapshot.display_is_held);
  EXPECT_TRUE(std::abs(snapshot.accepted_bpm - 72.33) < 1.0);
  EXPECT_EQ(snapshot.selected_method, "CONSENSUS");
  EXPECT_TRUE(snapshot.vqa.score_0_10 >= 8.0);
  EXPECT_TRUE(snapshot.router_shadow.active);
  EXPECT_TRUE(!snapshot.router_shadow.model_loaded);
  EXPECT_EQ(snapshot.router_shadow.backend, "heuristic_shadow");
  EXPECT_TRUE(!snapshot.router_shadow.recommended_method.empty());
  EXPECT_EQ(snapshot.consensus_methods.size(), 3U);
  EXPECT_TRUE(snapshot.consensus_spread_bpm <= 3.0);
}

void corrects_half_frequency_artifacts_like_python_consensus() {
  rppg_qnn::MeasurementEngine engine;
  const auto snapshot = engine.evaluate(
      {candidate("POS", 72.0, 0.83), candidate("CHROM", 144.0, 0.81),
       candidate("LGI", 73.0, 0.70)},
      good_quality(), 10.0);
  EXPECT_TRUE(snapshot.gate.accepted);
  EXPECT_TRUE(std::abs(snapshot.accepted_bpm - 72.5) < 1.0);
  EXPECT_TRUE(snapshot.consensus_artifact_correction.find("CHROM:halved") !=
              std::string::npos);
  EXPECT_EQ(snapshot.consensus_methods.size(), 3U);
}

void rejects_bad_quality_then_holds_for_five_seconds() {
  rppg_qnn::MeasurementEngine engine;
  (void)engine.evaluate(
      {candidate("POS", 72.0), candidate("CHROM", 73.0),
       candidate("LGI", 71.0)},
      good_quality(), 10.0);

  auto poor = good_quality();
  poor.brightness = 0.02;
  const auto held = engine.evaluate(
      {candidate("POS", 91.0), candidate("CHROM", 90.0)}, poor, 14.9);
  EXPECT_TRUE(!held.gate.accepted);
  EXPECT_TRUE(held.display_available);
  EXPECT_TRUE(held.display_is_held);
  EXPECT_TRUE(std::abs(held.display_bpm - 72.0) < 2.0);
  EXPECT_TRUE(!held.accepted_available);

  const auto expired = engine.evaluate(
      {candidate("POS", 91.0), candidate("CHROM", 90.0)}, poor, 15.1);
  EXPECT_TRUE(!expired.gate.accepted);
  EXPECT_TRUE(!expired.display_available);
  EXPECT_TRUE(!expired.display_is_held);
}

void rejects_method_disagreement_and_bpm_jump() {
  rppg_qnn::MeasurementEngine engine;
  (void)engine.evaluate(
      {candidate("POS", 72.0), candidate("CHROM", 73.0),
       candidate("LGI", 71.0)},
      good_quality(), 10.0);
  const auto disagreement = engine.evaluate(
      {candidate("POS", 110.0), candidate("CHROM", 140.0),
       candidate("LGI", 170.0)},
      good_quality(), 11.0);
  EXPECT_TRUE(!disagreement.gate.accepted);
  EXPECT_TRUE(!disagreement.gate.flags.empty());
}

}  // namespace

int main() {
  accepts_consensus_and_exposes_shadow_router();
  rejects_bad_quality_then_holds_for_five_seconds();
  rejects_method_disagreement_and_bpm_jump();
  corrects_half_frequency_artifacts_like_python_consensus();
  return test_support::finish();
}
