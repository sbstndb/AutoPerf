#pragma once

#include <cstddef>
#include <vector>

namespace autoperf {

// In-place AXPY: y = a * x + y
void axpy(float a, const std::vector<float>& x, std::vector<float>& y);

}  // namespace autoperf


