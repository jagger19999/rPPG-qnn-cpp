#pragma once

#include "rppg_qnn/deep_model.hpp"
#include "rppg_qnn/deep_runtime.hpp"

#include <memory>
#include <string>

namespace rppg_qnn::android {

struct OrtThreadOptions {
  int intra_op_threads{2};
  int inter_op_threads{1};
};

constexpr bool valid_ort_thread_options(OrtThreadOptions options) {
  return (options.intra_op_threads == 2 || options.intra_op_threads == 4 ||
          options.intra_op_threads == 6) &&
         options.inter_op_threads == 1;
}

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    DeepModel model, const std::string& model_path);
std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    DeepModel model, const std::string& model_path, OrtThreadOptions options);

}  // namespace rppg_qnn::android
