#include "kernel_matmul.hpp"

#include <cstddef>
#include <stdexcept>

namespace autoperf {

// Intentionally naive loop order (i,j,k) for baseline
void matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                std::size_t M,
                std::size_t N,
                std::size_t K) {
  if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
    throw std::invalid_argument("matmul: size mismatch");
  }
  
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

}  // namespace autoperf
