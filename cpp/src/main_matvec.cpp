#include "kernel_matvec.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
  constexpr std::size_t rows = 256;
  constexpr std::size_t cols = 1024;
  std::vector<float> A(rows * cols), x(cols), y(rows, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < rows * cols; ++i) A[i] = dist(rng);
  for (std::size_t j = 0; j < cols; ++j) x[j] = dist(rng);

  auto start = std::chrono::high_resolution_clock::now();
  autoperf::matvec(A, x, y, rows, cols);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = end - start;
  std::cout << "matvec completed in " << ms.count() << " ms\n";

  double checksum = 0.0;
  for (float v : y) checksum += v;
  std::cout << "checksum: " << checksum << "\n";
  return 0;
}
