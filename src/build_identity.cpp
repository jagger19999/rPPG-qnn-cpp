#include "rppg_qnn/build_identity.hpp"

namespace rppg_qnn {
namespace {

constexpr const char* platform_name() {
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

constexpr const char* abi_name() {
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

constexpr const char* camera_backend_name() {
#if defined(__ANDROID__) && defined(RPPG_ANDROID_CAMERA2)
  return "camera2";
#else
  return "not_compiled";
#endif
}

constexpr const char* deep_backend_name() {
#if defined(__ANDROID__) && defined(RPPG_ANDROID_ONNX_CPU)
  return "onnxruntime_cpu";
#else
  return "disabled";
#endif
}

}  // namespace

BuildIdentity build_identity() {
  return BuildIdentity{
      platform_name(), abi_name(), camera_backend_name(), deep_backend_name(),
      false};
}

std::string build_identity_text() {
  const BuildIdentity identity = build_identity();
  return "platform=" + identity.platform + ";abi=" + identity.abi +
         ";camera=" + identity.camera_backend + ";deep=" +
         identity.deep_backend + ";qnn_ready=" +
         (identity.qnn_ready ? "true" : "false");
}

}  // namespace rppg_qnn
