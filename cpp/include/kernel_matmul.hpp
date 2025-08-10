#pragma once

#include <cstddef>
#include <vector>

namespace autoperf {

// Naive matrix-matrix product: C = A * B
// A (M, K), B (K, N), C (M, N) are in row-major layout.
void matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                std::size_t M,
                std::size_t N,
                std::size_t K);

}
