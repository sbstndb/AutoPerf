#include <cstddef>
#include <immintrin.h>

void matmul(const float* A, const float* B, float* C, size_t N) {
    const size_t block_i = 96;
    const size_t block_k = 128;

    for (size_t i_tile = 0; i_tile < N; i_tile += block_i) {
        size_t i_limit = (i_tile + block_i > N) ? N : i_tile + block_i;

        for (size_t k_tile = 0; k_tile < N; k_tile += block_k) {
            size_t k_limit = (k_tile + block_k > N) ? N : k_tile + block_k;

            for (size_t i = i_tile; i < i_limit; i += 6) {
                if (i + 6 > i_limit) {
                    // Row remainder handling
                    for (size_t ir = i; ir < i_limit; ++ir) {
                        float* c_ptr = &C[ir * N];
                        for (size_t j = 0; j + 7 < N; j += 8) {
                            __m256 vc = (k_tile == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(c_ptr + j);
                            for (size_t k = k_tile; k < k_limit; ++k) {
                                vc = _mm256_fmadd_ps(_mm256_set1_ps(A[ir * N + k]), _mm256_loadu_ps(&B[k * N + j]), vc);
                            }
                            _mm256_storeu_ps(c_ptr + j, vc);
                        }
                        for (size_t j = (N & ~7); j < N; ++j) {
                            float sum = (k_tile == 0) ? 0.0f : c_ptr[j];
                            for (size_t k = k_tile; k < k_limit; ++k) {
                                sum += A[ir * N + k] * B[k * N + j];
                            }
                            c_ptr[j] = sum;
                        }
                    }
                    continue;
                }

                size_t j = 0;
                for (; j + 15 < N; j += 16) {
                    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

                    if (k_tile == 0) {
                        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
                    } else {
                        c00 = _mm256_loadu_ps(&C[(i + 0) * N + j]); c01 = _mm256_loadu_ps(&C[(i + 0) * N + j + 8]);
                        c10 = _mm256_loadu_ps(&C[(i + 1) * N + j]); c11 = _mm256_loadu_ps(&C[(i + 1) * N + j + 8]);
                        c20 = _mm256_loadu_ps(&C[(i + 2) * N + j]); c21 = _mm256_loadu_ps(&C[(i + 2) * N + j + 8]);
                        c30 = _mm256_loadu_ps(&C[(i + 3) * N + j]); c31 = _mm256_loadu_ps(&C[(i + 3) * N + j + 8]);
                        c40 = _mm256_loadu_ps(&C[(i + 4) * N + j]); c41 = _mm256_loadu_ps(&C[(i + 4) * N + j + 8]);
                        c50 = _mm256_loadu_ps(&C[(i + 5) * N + j]); c51 = _mm256_loadu_ps(&C[(i + 5) * N + j + 8]);
                    }

                    for (size_t k = k_tile; k < k_limit; ++k) {
                        __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
                        __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);

                        c00 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 0) * N + k]), b0, c00);
                        c01 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 0) * N + k]), b1, c01);
                        c10 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 1) * N + k]), b0, c10);
                        c11 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 1) * N + k]), b1, c11);
                        c20 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 2) * N + k]), b0, c20);
                        c21 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 2) * N + k]), b1, c21);
                        c30 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 3) * N + k]), b0, c30);
                        c31 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 3) * N + k]), b1, c31);
                        c40 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 4) * N + k]), b0, c40);
                        c41 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 4) * N + k]), b1, c41);
                        c50 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 5) * N + k]), b0, c50);
                        c51 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 5) * N + k]), b1, c51);
                    }

                    _mm256_storeu_ps(&C[(i + 0) * N + j], c00); _mm256_storeu_ps(&C[(i + 0) * N + j + 8], c01);
                    _mm256_storeu_ps(&C[(i + 1) * N + j], c10); _mm256_storeu_ps(&C[(i + 1) * N + j + 8], c11);
                    _mm256_storeu_ps(&C[(i + 2) * N + j], c20); _mm256_storeu_ps(&C[(i + 2) * N + j + 8], c21);
                    _mm256_storeu_ps(&C[(i + 3) * N + j], c30); _mm256_storeu_ps(&C[(i + 3) * N + j + 8], c31);
                    _mm256_storeu_ps(&C[(i + 4) * N + j], c40); _mm256_storeu_ps(&C[(i + 4) * N + j + 8], c41);
                    _mm256_storeu_ps(&C[(i + 5) * N + j], c50); _mm256_storeu_ps(&C[(i + 5) * N + j + 8], c51);
                }

                for (; j < N; ++j) {
                    for (size_t r = 0; r < 6; ++r) {
                        float sum = (k_tile == 0) ? 0.0f : C[(i + r) * N + j];
                        for (size_t k = k_tile; k < k_limit; ++k) {
                            sum += A[(i + r) * N + k] * B[k * N + j];
                        }
                        C[(i + r) * N + j] = sum;
                    }
                }
            }
        }
    }
}