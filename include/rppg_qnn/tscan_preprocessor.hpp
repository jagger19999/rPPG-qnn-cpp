#pragma once

#include "rppg_qnn/deep_runtime.hpp"

#include <cstdint>
#include <vector>

namespace rppg_qnn {

struct TscanTensor {
  std::vector<float> values;
  std::vector<std::int64_t> shape;
};

TscanTensor preprocess_tscan_rgb(const DeepInput& input);

}  // namespace rppg_qnn
