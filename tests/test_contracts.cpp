#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/error.hpp"

#include <string>

#include "test_support.hpp"

int main() {
  const rppg_qnn::HeartRateResult result;

  EXPECT_EQ(result.schema_version, 1);
  EXPECT_TRUE(result.method.empty());

  rppg_qnn::FrameHealth health{};
  EXPECT_EQ(health.schema_version, 1);
  EXPECT_EQ(rppg_qnn::to_string(rppg_qnn::ErrorCode::QnnGpuInitFailed),
            "QNN_GPU_INIT_FAILED");
  rppg_qnn::AppError error(rppg_qnn::ErrorCode::ConfigInvalid,
                           "camera and video conflict");
  EXPECT_EQ(error.code(), rppg_qnn::ErrorCode::ConfigInvalid);
  EXPECT_TRUE(std::string(error.what()).find("camera and video conflict") !=
              std::string::npos);

  return test_support::finish();
}
