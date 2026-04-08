#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* const end = data + size;
    const __m256i target = _mm256_set1_epi32(value);

    // Process 16 elements per iteration (2 x 256-bit vectors)
    // This unroll factor is optimized for the Intel Ultra 7's branch predictor and ILP
    while (end - ptr >= 16) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(ptr + 8));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);

        // Combine masks using OR
        __m256i combined = _mm256_or_si256(c0, c1);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(combined));

        if (mask != 0) {
            // Match found in this 16-element block
            int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            if (m0 != 0) {
                return (int)(ptr - data) + __builtin_ctz(m0);
            }
            int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            return (int)(ptr - data) + 8 + __builtin_ctz(m1);
        }
        ptr += 16;
    }

    // Process remaining 8-element chunk
    if (end - ptr >= 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) {
            return (int)(ptr - data) + __builtin_ctz(mask);
        }
        ptr += 8;
    }

    // Scalar tail for remaining < 8 elements
    while (ptr < end) {
        if (*ptr == value) {
            return (int)(ptr - data);
        }
        ptr++;
    }

    return -1;
}