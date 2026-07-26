#include "rppg_qnn/waveform_snapshot.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "test_support.hpp"

namespace {

void normalizes_finite_waveform() {
  const auto snapshot = rppg_qnn::make_waveform_snapshot(
      {2.0F, 3.0F, 4.0F}, "POS", 30.0, true, "");
  EXPECT_TRUE(snapshot.available);
  EXPECT_EQ(snapshot.values.size(), std::size_t{3});
  EXPECT_TRUE(std::abs(snapshot.values[0] + 1.0F) < 1e-6F);
  EXPECT_TRUE(std::abs(snapshot.values[1]) < 1e-6F);
  EXPECT_TRUE(std::abs(snapshot.values[2] - 1.0F) < 1e-6F);
  EXPECT_EQ(snapshot.method, std::string("POS"));
  EXPECT_EQ(snapshot.sample_rate_hz, 30.0);
  EXPECT_TRUE(snapshot.is_valid);
}

void rejects_constant_or_non_finite_waveform() {
  EXPECT_TRUE(!rppg_qnn::make_waveform_snapshot(
                   {1.0F, 1.0F, 1.0F}, "GREEN", 30.0, false, "flat")
                   .available);
  EXPECT_TRUE(!rppg_qnn::make_waveform_snapshot(
                   {0.0F, std::numeric_limits<float>::infinity()}, "TSCAN",
                   30.0, false, "waveform_invalid")
                   .available);
}

void selects_python_compatible_sample_rates() {
  rppg_qnn::HeartRateResult traditional;
  traditional.method = "CHROM";
  traditional.source_fps = 21.5;
  EXPECT_EQ(rppg_qnn::waveform_sample_rate_hz(traditional), 21.5);

  rppg_qnn::HeartRateResult deep;
  deep.method = "TSCAN";
  deep.source_fps = 21.5;
  EXPECT_EQ(rppg_qnn::waveform_sample_rate_hz(deep), 30.0);
}

}  // namespace

int main() {
  normalizes_finite_waveform();
  rejects_constant_or_non_finite_waveform();
  selects_python_compatible_sample_rates();
  return test_support::finish();
}
