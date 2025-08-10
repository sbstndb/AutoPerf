#include "kernel_matvec.hpp"

namespace autoperf {

void matvec(const std::vector<float>& A,
                const std::vector<float>& x,
                std::vector<float>& y,
                std::size_t rows,
                std::size_t cols) {
  for (std::size_t i = 0; i < rows; ++i) {
    float acc = 0.0f;
    for (std::size_t j = 0; j < cols; ++j) {
      acc += A[i * cols + j] * x[j];
    }
    y[i] = acc;
  }
}

} // namespace autoperf
