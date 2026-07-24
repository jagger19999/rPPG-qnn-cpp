#pragma once

#include "rppg_qnn/deep_runtime.hpp"

#include <memory>
#include <string>

namespace rppg_qnn::android {

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    const std::string& model_path);

}  // namespace rppg_qnn::android
