#include <gtest/gtest.h>

#include "kernel_matvec.hpp"

#include <random>

static std::vector<float> matvec_ref(const std::vector<float>& A,
                                     const std::vector<float>& x,
                                     std::size_t rows,
                                     std::size_t cols) {
  std::vector<float> y(rows, 0.0f);
  for (std::size_t i = 0; i < rows; ++i) {
    float acc = 0.0f;
    for (std::size_t j = 0; j < cols; ++j) {
      acc += A[i * cols + j] * x[j];
    }
    y[i] = acc;
  }
  return y;
}

TEST(MatVec, SmallKnown) {
  constexpr std::size_t rows = 2, cols = 3;
  std::vector<float> A = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  std::vector<float> x = {1.0f, -1.0f, 2.0f};
  std::vector<float> y(rows, 0.0f);
  autoperf::matvec(A, x, y, rows, cols);
  auto ref = matvec_ref(A, x, rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    EXPECT_NEAR(y[i], ref[i], 1e-3f);
  }
}

TEST(MatVec, Random) {
  constexpr std::size_t rows = 64, cols = 128;
  std::vector<float> A(rows * cols), x(cols), y(rows, 0.0f);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (std::size_t i = 0; i < rows * cols; ++i) A[i] = dist(rng);
  for (std::size_t j = 0; j < cols; ++j) x[j] = dist(rng);
  autoperf::matvec(A, x, y, rows, cols);
  auto ref = matvec_ref(A, x, rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    EXPECT_NEAR(y[i], ref[i], 1e-3f);
  }
}


