#pragma once

#include "rppg_qnn/config.hpp"
#include "rppg_qnn/contracts.hpp"

#include <string>
#include <vector>

namespace rppg_qnn {

struct LibraryProbe {
  bool loaded{false};
  std::string resolved_path;
  std::string error;
};

LibraryProbe probe_library(const std::string& path,
                           const std::vector<std::string>& required_symbols);

PreflightResult run_qnn_preflight(const AppConfig& config);

}  // namespace rppg_qnn
