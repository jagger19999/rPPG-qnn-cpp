#include "rppg_qnn/build_identity.hpp"

#include <string>

#include "test_support.hpp"

namespace {

constexpr const char* expected_platform() {
#if defined(__ANDROID__)
  return "android";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

constexpr const char* expected_abi() {
#if defined(__ANDROID__) && \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
  return "arm64-v8a";
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#else
  return "unknown";
#endif
}

void build_identity_has_foundation_fields() {
  const auto identity = rppg_qnn::build_identity();

  EXPECT_EQ(identity.platform, expected_platform());
  EXPECT_EQ(identity.abi, expected_abi());
  EXPECT_EQ(identity.camera_backend, "not_compiled");
  EXPECT_EQ(identity.deep_backend, "disabled");
  EXPECT_EQ(identity.qnn_ready, false);
}

void build_identity_text_is_stable() {
  const auto identity = rppg_qnn::build_identity();
  const std::string text = rppg_qnn::build_identity_text();

  EXPECT_TRUE(text.find("platform=") != std::string::npos);
  EXPECT_TRUE(text.find("abi=") != std::string::npos);
  EXPECT_TRUE(text.find("camera=not_compiled") != std::string::npos);
  EXPECT_TRUE(text.find("deep=disabled") != std::string::npos);
  EXPECT_TRUE(text.find("qnn_ready=false") != std::string::npos);

  const std::string expected =
      "platform=" + identity.platform + ";abi=" + identity.abi +
      ";camera=" + identity.camera_backend + ";deep=" +
      identity.deep_backend + ";qnn_ready=" +
      (identity.qnn_ready ? "true" : "false");
  EXPECT_EQ(text, expected);
}

}  // namespace

int main() {
  build_identity_has_foundation_fields();
  build_identity_text_is_stable();
  return test_support::finish();
}
