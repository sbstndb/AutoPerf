#include "kernel_axpy.hpp"
#include <stdexcept>

namespace autoperf {

void axpy(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t n = x.size();
  if (y.size() != n) {
    throw std::invalid_argument("axpy: size mismatch");
  }
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

}  // namespace autoperf


