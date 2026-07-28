#include "rppg_qnn/efficientphys_postprocessor.hpp"

#include "rppg_qnn/signal_processing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rppg_qnn {
namespace {

// 1st-order Butterworth band-pass [0.6, 3.3] Hz at FS=30.
// Coefficients match scipy.signal.butter(1, [0.6/30*2, 3.3/30*2], 'bandpass').
// zi matches scipy.signal.lfilter_zi(b, a).
const IirFilter& efficientphys_bandpass_filter() {
  static const IirFilter filter{
      {0.22512267390361498, 0.0, -0.22512267390361498},
      {1.0, -1.4811036685329875, 0.54975465219277},
      {-0.22512267390361554, -0.2251226739036147}};
  return filter;
}

}  // namespace

EfficientPhysPostprocessResult reconstruct_efficientphys_bvp(
    const std::vector<float>& raw_diff) {
  EfficientPhysPostprocessResult result;

  // Input contract: exactly 180 finite floats.
  if (raw_diff.size() != 180U ||
      !std::all_of(raw_diff.begin(), raw_diff.end(),
                   [](float value) { return std::isfinite(value); })) {
    result.invalid_reason = "EFFICIENTPHYS_RAW_DIFF_INVALID";
    return result;
  }

  // Step 1: cumsum (np.cumsum)
  std::vector<double> cumsummed(180);
  {
    double accumulator = 0.0;
    for (std::size_t i = 0; i < 180U; ++i) {
      accumulator += static_cast<double>(raw_diff[i]);
      cumsummed[i] = accumulator;
    }
  }

  // Step 2: smoothness-prior detrend (lambda = 100)
  std::vector<double> detrended = smoothness_priors_detrend(cumsummed);

  // Step 3 + 4: Butterworth band-pass [0.6, 3.3] Hz + filtfilt
  std::vector<double> filtered =
      filtfilt(detrended, efficientphys_bandpass_filter());

  // Convert to float32 output
  result.bvp.reserve(180);
  for (double value : filtered) {
    if (!std::isfinite(value)) {
      result.invalid_reason = "EFFICIENTPHYS_RECONSTRUCTION_NON_FINITE";
      return result;
    }
    result.bvp.push_back(static_cast<float>(value));
  }

  result.is_valid = true;
  return result;
}

}  // namespace rppg_qnn
