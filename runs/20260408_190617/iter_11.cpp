#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // Process 64 elements per iteration (8 x 8-lane YMM registers)
    // This maximizes throughput on Intel Ultra's wide execution engine
    while (end - ptr >= 64) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)(ptr));
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

        // Combine masks into two 32-bit integers (each representing 32 elements)
        // movemask_ps extracts 1 bit per 32-bit lane
        int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
        int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
        int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
        int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
        int mask_low = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);

        int m4 = _mm256_movemask_ps(_mm256_castsi256_ps(c4));
        int m5 = _mm256_movemask_ps(_mm256_castsi256_ps(c5));
        int m6 = _mm256_movemask_ps(_mm256_castsi256_ps(c6));
        int m7 = _mm256_movemask_ps(_mm256_castsi256_ps(c7));
        int mask_high = m4 | (m5 << 8) | (m6 << 16) | (m7 << 24);

        if (mask_low != 0) {
            return (int)(ptr - data) + __builtin_ctz(mask_low);
        }
        if (mask_high != 0) {
            return (int)(ptr - data) + 32 + __builtin_ctz(mask_high);
        }
        ptr += 64;
    }

    // Process remaining in 8-element steps
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