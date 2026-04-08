#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // SIMD Path: AVX2
    // Process 8 integers at a time, unrolled by 4 for better ILP
    if (size >= 8) {
        __m256i target = _mm256_set1_epi32(value);
        
        // Unroll to 32 elements per iteration
        for (; i <= size - 32; i += 32) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 16));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 24));

            __m256i cmp0 = _mm256_cmpeq_epi32(v0, target);
            __m256i cmp1 = _mm256_cmpeq_epi32(v1, target);
            __m256i cmp2 = _mm256_cmpeq_epi32(v2, target);
            __m256i cmp3 = _mm256_cmpeq_epi32(v3, target);

            // Combine results to check if any match exists in this block of 32
            __m256i or01 = _mm256_or_si256(cmp0, cmp1);
            __m256i or23 = _mm256_or_si256(cmp2, cmp3);
            __m256i combined = _mm256_or_si256(or01, or23);

            if (!_mm256_testz_si256(combined, combined)) {
                // Match found in this block, find exactly where
                auto check_match = [&](__m256i cmp, int offset) {
                    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                    if (mask != 0) {
                        return i + offset + __builtin_ctz(mask);
                    }
                    return -1;
                };

                int res = check_match(cmp0, 0);
                if (res != -1) return res;
                res = check_match(cmp1, 8);
                if (res != -1) return res;
                res = check_match(cmp2, 16);
                if (res != -1) return res;
                return check_match(cmp3, 24);
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

    // Scalar tail
    for (; i < size; i++) {
        if (data[i] == value)
            return i;
    }

    return -1;
}