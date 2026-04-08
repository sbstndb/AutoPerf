#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;

    // Vectorized search: process 32 elements at a time (4 x 8-way AVX2)
    if (remaining >= 32) {
        __m256i target = _mm256_set1_epi32(value);
        
        while (remaining >= 32) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 8));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 16));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 24));

            __m256i cmp0 = _mm256_cmpeq_epi32(v0, target);
            __m256i cmp1 = _mm256_cmpeq_epi32(v1, target);
            __m256i cmp2 = _mm256_cmpeq_epi32(v2, target);
            __m256i cmp3 = _mm256_cmpeq_epi32(v3, target);

            // Combine masks to check if any match exists in this 32-element block
            __m256i or01 = _mm256_or_si256(cmp0, cmp1);
            __m256i or23 = _mm256_or_si256(cmp2, cmp3);
            __m256i combined = _mm256_or_si256(or01, or23);

            if (!_mm256_testz_si256(combined, combined)) {
                // Match found in this block, find exactly where
                int mask;
                if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp0))))
                    return (int)(ptr - data) + __builtin_ctz(mask);
                if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp1))))
                    return (int)(ptr - data) + 8 + __builtin_ctz(mask);
                if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp2))))
                    return (int)(ptr - data) + 16 + __builtin_ctz(mask);
                mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp3));
                return (int)(ptr - data) + 24 + __builtin_ctz(mask);
            }

            ptr += 32;
            remaining -= 32;
        }
    }

    // Scalar tail handling (also handles small arrays)
    for (int i = 0; i < remaining; i++) {
        if (ptr[i] == value)
            return (int)(ptr - data) + i;
    }

    return -1;
}