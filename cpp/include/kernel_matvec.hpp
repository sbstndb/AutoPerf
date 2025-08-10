#pragma once

#include <cstddef>
#include <vector>

namespace autoperf {

// Naive matrix-vector product: y = A * x
// A is in row-major layout with size rows * cols.
// Precondition: A.size() == rows * cols, x.size() == cols, y.size() == rows.
void matvec(const std::vector<float>& A,
                const std::vector<float>& x,
                std::vector<float>& y,
                std::size_t rows,
                std::size_t cols);

}


