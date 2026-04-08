#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    if (size >= 8) {
        const __m256i target = _mm256_set1_epi32(value);
        
        // Unroll by 64 bytes (16 integers) to saturate load ports and reduce branches
        for (; i <= size - 16; i += 16) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));

            __m256i cmp0 = _mm256_cmpeq_epi32(v0, target);
            __m256i cmp1 = _mm256_cmpeq_epi32(v1, target);

            int mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp0));
            int mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp1));

            if (mask0 != 0) {
                return i + __builtin_ctz(mask0);
            }
            if (mask1 != 0) {
                return i + 8 + __builtin_ctz(mask1);
            }
        }

        // Handle remaining 8-element chunks
        for (; i <= size - 8; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            __m256i cmp = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask != 0) {
                return i + __builtin_ctz(mask);
            }
        }
    }

    // Scalar tail for remaining elements
    for (; i < size; i++) {
        if (data[i] == value)
            return i;
    }

    return -1;
}