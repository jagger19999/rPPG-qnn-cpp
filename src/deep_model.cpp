#include "rppg_qnn/deep_model.hpp"

#include "rppg_qnn/error.hpp"

#include <string>

namespace rppg_qnn {

DeepModel parse_deep_model(std::string_view value) {
  if (value == "disabled") {
    return DeepModel::Disabled;
  }
  if (value == "tscan") {
    return DeepModel::Tscan;
  }
  if (value == "efficientphys") {
    return DeepModel::EfficientPhys;
  }
  throw AppError(ErrorCode::ConfigInvalid,
                 "deep model must be disabled, tscan, or efficientphys; got '" +
                     std::string(value) + "'");
}

std::string_view to_string(DeepModel model) {
  switch (model) {
    case DeepModel::Disabled:
      return "disabled";
    case DeepModel::Tscan:
      return "tscan";
    case DeepModel::EfficientPhys:
      return "efficientphys";
  }
  throw AppError(ErrorCode::ConfigInvalid,
                 "deep model enum value is invalid: " +
                     std::to_string(static_cast<int>(model)));
}

}  // namespace rppg_qnn
