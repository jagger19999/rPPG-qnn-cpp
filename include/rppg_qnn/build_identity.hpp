#pragma once

#include <string>

namespace rppg_qnn {

struct BuildIdentity {
  std::string platform;
  std::string abi;
  std::string camera_backend;
  std::string deep_backend;
  bool qnn_ready{false};
};

[[nodiscard]] BuildIdentity build_identity();
[[nodiscard]] std::string build_identity_text();

}  // namespace rppg_qnn
