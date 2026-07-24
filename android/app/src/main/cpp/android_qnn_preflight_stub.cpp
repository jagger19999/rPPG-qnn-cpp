#include "rppg_qnn/qnn_preflight.hpp"

namespace rppg_qnn {

PreflightResult run_qnn_preflight(const AppConfig& config) {
  PreflightResult result;
  if (config.deep != "disabled") {
    result.error = "deep runtime is not configured in this Android build";
  }
  return result;
}

}  // namespace rppg_qnn
