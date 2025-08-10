#include <gtest/gtest.h>

#include "kernel_matmul.hpp"

#include <random>

static std::vector<float> matmul_ref(const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     std::size_t M,
                                     std::size_t N,
                                     std::size_t K) {
  std::vector<float> C(M * N, 0.0f);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
  return C;
}

TEST(MatMul, SmallKnown) {
  constexpr std::size_t M = 2, N = 2, K = 2;
  std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> C(M * N, 0.0f);
  
  autoperf::matmul(A, B, C, M, N, K);
  auto ref = matmul_ref(A, B, M, N, K);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(C[i], ref[i], 1e-3f);
  }
}

TEST(MatMul, Random) {
  constexpr std::size_t M = 32, N = 32, K = 32;
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  for (std::size_t i = 0; i < M * K; ++i) A[i] = dist(rng);
  for (std::size_t i = 0; i < K * N; ++i) B[i] = dist(rng);
  
  autoperf::matmul(A, B, C, M, N, K);
  auto ref = matmul_ref(A, B, M, N, K);

  for (std::size_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(C[i], ref[i], 1e-3f);
  }
}
