#include "kernel_matmul.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
  constexpr std::size_t M = 128, N = 128, K = 128;
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < M * K; ++i) A[i] = dist(rng);
  for (std::size_t i = 0; i < K * N; ++i) B[i] = dist(rng);

  auto start = std::chrono::high_resolution_clock::now();
  autoperf::matmul(A, B, C, M, N, K);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = end - start;
  std::cout << "matmul completed in " << ms.count() << " ms\n";

  double checksum = 0.0;
  for (float v : C) checksum += v;
  std::cout << "checksum: " << checksum << "\n";
  return 0;
}
