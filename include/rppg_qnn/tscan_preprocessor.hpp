#pragma once

#include "rppg_qnn/deep_runtime.hpp"

#include <cstdint>
#include <vector>

namespace rppg_qnn {

struct TscanTensor {
  std::vector<float> values;
  std::vector<std::int64_t> shape;
};

// Own one instance per deep worker. The returned tensor remains valid until
// the next preprocess call on this instance.
class TscanPreprocessor {
 public:
  TscanPreprocessor();

  const TscanTensor& preprocess(const DeepInput& input);

 private:
  std::vector<float> differences_;
  TscanTensor output_;
};

TscanTensor preprocess_tscan_rgb(const DeepInput& input);

}  // namespace rppg_qnn
