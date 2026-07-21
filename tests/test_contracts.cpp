#include "rppg_qnn/contracts.hpp"

#include "test_support.hpp"

int main() {
  const rppg_qnn::HeartRateResult result;

  EXPECT_EQ(result.schema_version, 1);
  EXPECT_TRUE(result.method.empty());

  return test_support::finish();
}
