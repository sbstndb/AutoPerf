#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // Process 64 elements per iteration (8 x 256-bit vectors)
    // This maximizes ILP on the Intel Ultra 7 (Meteor Lake) architecture
    while (end - ptr >= 64) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(ptr + 8));
        __m256i r2 = _mm256_loadu_si256((const __m256i*)(ptr + 16));
        __m256i r3 = _mm256_loadu_si256((const __m256i*)(ptr + 24));
        __m256i r4 = _mm256_loadu_si256((const __m256i*)(ptr + 32));
        __m256i r5 = _mm256_loadu_si256((const __m256i*)(ptr + 40));
        __m256i r6 = _mm256_loadu_si256((const __m256i*)(ptr + 48));
        __m256i r7 = _mm256_loadu_si256((const __m256i*)(ptr + 56));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);
        __m256i c4 = _mm256_cmpeq_epi32(r4, target);
        __m256i c5 = _mm256_cmpeq_epi32(r5, target);
        __m256i c6 = _mm256_cmpeq_epi32(r6, target);
        __m256i c7 = _mm256_cmpeq_epi32(r7, target);

        __m256i m01 = _mm256_or_si256(c0, c1);
        __m256i m23 = _mm256_or_si256(c2, c3);
        __m256i m45 = _mm256_or_si256(c4, c5);
        __m256i m67 = _mm256_or_si256(c6, c7);
        
        __m256i m03 = _mm256_or_si256(m01, m23);
        __m256i m47 = _mm256_or_si256(m45, m67);
        __m256i combined = _mm256_or_si256(m03, m47);

        if (!_mm256_testz_si256(combined, combined)) {
            // Match found: check individual masks
            auto check_mask = [&](__m256i c, int offset) -> int {
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(c));
                return m ? (int)(ptr - data) + offset + __builtin_ctz(m) : -1;
            };

            int res;
            if ((res = check_mask(c0, 0))  != -1) return res;
            if ((res = check_mask(c1, 8))  != -1) return res;
            if ((res = check_mask(c2, 16)) != -1) return res;
            if ((res = check_mask(c3, 24)) != -1) return res;
            if ((res = check_mask(c4, 32)) != -1) return res;
            if ((res = check_mask(c5, 40)) != -1) return res;
            if ((res = check_mask(c6, 48)) != -1) return res;
            return check_mask(c7, 56);
        }
        ptr += 64;
    }

    // Process remaining 8-element chunks
    while (end - ptr >= 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) {
            return (int)(ptr - data) + __builtin_ctz(mask);
        }
        ptr += 8;
    }

    // Final scalar tail
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}