#pragma once

#include "rppg_qnn/traditional_predictor.hpp"

namespace rppg_qnn {

class GreenPredictor final : public TraditionalPredictor {
 public:
  GreenPredictor() : TraditionalPredictor(TraditionalMethod::Green) {}
};

}  // namespace rppg_qnn
