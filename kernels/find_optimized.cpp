#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    __m256i target = _mm256_set1_epi32(value);

    // Unroll to 64 elements (8 vectors) to maximize ILP and saturate load ports
    for (; i <= size - 64; i += 64) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(data + i + 8));
        __m256i r2 = _mm256_loadu_si256((const __m256i*)(data + i + 16));
        __m256i r3 = _mm256_loadu_si256((const __m256i*)(data + i + 24));
        __m256i r4 = _mm256_loadu_si256((const __m256i*)(data + i + 32));
        __m256i r5 = _mm256_loadu_si256((const __m256i*)(data + i + 40));
        __m256i r6 = _mm256_loadu_si256((const __m256i*)(data + i + 48));
        __m256i r7 = _mm256_loadu_si256((const __m256i*)(data + i + 56));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);
        __m256i c4 = _mm256_cmpeq_epi32(r4, target);
        __m256i c5 = _mm256_cmpeq_epi32(r5, target);
        __m256i c6 = _mm256_cmpeq_epi32(r6, target);
        __m256i c7 = _mm256_cmpeq_epi32(r7, target);

        __m256i or01 = _mm256_or_si256(c0, c1);
        __m256i or23 = _mm256_or_si256(c2, c3);
        __m256i or45 = _mm256_or_si256(c4, c5);
        __m256i or67 = _mm256_or_si256(c6, c7);
        
        __m256i or_all = _mm256_or_si256(_mm256_or_si256(or01, or23), _mm256_or_si256(or45, or67));

        if (!_mm256_testz_si256(or_all, or_all)) {
            // Match found in this 64-element block
            auto check_mask = [&](__m256i mask, int offset) {
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
                return (m != 0) ? (offset + __builtin_ctz(m)) : -1;
            };

            int res;
            if ((res = check_mask(c0, i)) != -1) return res;
            if ((res = check_mask(c1, i + 8)) != -1) return res;
            if ((res = check_mask(c2, i + 16)) != -1) return res;
            if ((res = check_mask(c3, i + 24)) != -1) return res;
            if ((res = check_mask(c4, i + 32)) != -1) return res;
            if ((res = check_mask(c5, i + 40)) != -1) return res;
            if ((res = check_mask(c6, i + 48)) != -1) return res;
            return check_mask(c7, i + 56);
        }
    }

    // Handle remaining 8-element chunks
    for (; i <= size - 8; i += 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) return i + __builtin_ctz(mask);
    }

    // Final scalar tail
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}