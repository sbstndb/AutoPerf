#include <gtest/gtest.h>
#include "kernel_custom.hpp"

TEST(CustomTest, BasicTest) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  float output = 0.0f;
  custom_operation(input, &output);
  // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
  EXPECT_NEAR(output, 14.0f, 1e-6f);
}

TEST(CustomTest, EmptyTest) {
  std::vector<float> input;
  float output = -1.0f;
  custom_operation(input, &output);
  EXPECT_NEAR(output, 0.0f, 1e-6f);
}
