#pragma once

#include <string>
#include <vector>

namespace rppg_qnn {

// Reconstructed BVP result from EfficientPhys differential output.
//
// EfficientPhys (DiffNormalized label type) produces a first-difference
// representation of the PPG signal.  The official rPPG-Toolbox post-processing
// pipeline reconstructs a smooth BVP waveform via:
//   1. np.cumsum(predictions)
//   2. smoothness-prior detrend (lambda = 100)
//   3. 1st-order Butterworth band-pass [0.6, 3.3] Hz at 30 FPS
//   4. scipy.signal.filtfilt (zero-phase, odd-extension padding)
//
// Reference: rPPG-Toolbox/evaluation/post_process.py
//            calculate_metric_per_video(diff_flag=True, use_bandpass=True)
struct EfficientPhysPostprocessResult {
  std::vector<float> bvp;          // 180-point reconstructed BVP
  bool is_valid{false};
  std::string invalid_reason;
};

// Reconstruct BVP from EfficientPhys raw differential output.
//
// \param raw_diff  180 finite float32 values from the ONNX pulse output.
// \return          Reconstructed BVP or invalid result.
EfficientPhysPostprocessResult reconstruct_efficientphys_bvp(
    const std::vector<float>& raw_diff);

}  // namespace rppg_qnn
