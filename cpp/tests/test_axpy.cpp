#include <gtest/gtest.h>
#include "kernel_axpy.hpp"
#include <random>

TEST(AXPY, SmallKnown) {
  std::vector<float> x = {1.f, -2.f, 3.f};
  std::vector<float> y = {4.f, 5.f, -6.f};
  autoperf::axpy(2.0f, x, y);
  EXPECT_NEAR(y[0], 2*1 + 4, 1e-6f);
  EXPECT_NEAR(y[1], 2*(-2) + 5, 1e-6f);
  EXPECT_NEAR(y[2], 2*3 + (-6), 1e-6f);
}

TEST(AXPY, RandomVsRef) {
  const std::size_t n = 1000;
  std::vector<float> x(n), y(n), y_ref(n);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
    y_ref[i] = y[i];
  }
  autoperf::axpy(2.0f, x, y);
  for (std::size_t i = 0; i < n; ++i) {
    y_ref[i] = 2.0f * x[i] + y_ref[i];
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i], y_ref[i], 1e-6f);
  }
}


