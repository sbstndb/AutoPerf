#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;

    if (remaining >= 8) {
        __m256i target = _mm256_set1_epi32(value);

        // Large unroll: 64 elements (8 registers) per iteration
        while (remaining >= 64) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 8));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 16));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 24));
            __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 32));
            __m256i v5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 40));
            __m256i v6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 48));
            __m256i v7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 56));

            __m256i c0 = _mm256_cmpeq_epi32(v0, target);
            __m256i c1 = _mm256_cmpeq_epi32(v1, target);
            __m256i c2 = _mm256_cmpeq_epi32(v2, target);
            __m256i c3 = _mm256_cmpeq_epi32(v3, target);
            __m256i c4 = _mm256_cmpeq_epi32(v4, target);
            __m256i c5 = _mm256_cmpeq_epi32(v5, target);
            __m256i c6 = _mm256_cmpeq_epi32(v6, target);
            __m256i c7 = _mm256_cmpeq_epi32(v7, target);

            __m256i or01 = _mm256_or_si256(c0, c1);
            __m256i or23 = _mm256_or_si256(c2, c3);
            __m256i or45 = _mm256_or_si256(c4, c5);
            __m256i or67 = _mm256_or_si256(c6, c7);
            
            __m256i or03 = _mm256_or_si256(or01, or23);
            __m256i or47 = _mm256_or_si256(or45, or67);
            __m256i combined = _mm256_or_si256(or03, or47);

            if (!_mm256_testz_si256(combined, combined)) {
                // Match found in this 64-element block
                auto check_mask = [&](__m256i cmp, int offset) -> int {
                    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                    if (mask) return (int)(ptr - data) + offset + __builtin_ctz(mask);
                    return -1;
                };

                int res;
                if ((res = check_mask(c0, 0)) != -1) return res;
                if ((res = check_mask(c1, 8)) != -1) return res;
                if ((res = check_mask(c2, 16)) != -1) return res;
                if ((res = check_mask(c3, 24)) != -1) return res;
                if ((res = check_mask(c4, 32)) != -1) return res;
                if ((res = check_mask(c5, 40)) != -1) return res;
                if ((res = check_mask(c6, 48)) != -1) return res;
                return check_mask(c7, 56);
            }
            ptr += 64;
            remaining -= 64;
        }

        // Mid-size loop: 8 elements per iteration
        while (remaining >= 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i cmp = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask) {
                return (int)(ptr - data) + __builtin_ctz(mask);
            }
            ptr += 8;
            remaining -= 8;
        }
    }

    // Final scalar tail
    for (int i = 0; i < remaining; i++) {
        if (ptr[i] == value)
            return (int)(ptr - data) + i;
    }

    return -1;
}