#include <cstddef>
#include <immintrin.h>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // 1. Small Preamble: Align ptr to 32-byte boundary
    while (((uintptr_t)ptr & 31) != 0 && ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    // 2. Main Loop: Process 16 elements per iteration (2 x 256-bit vectors)
    // Using aligned loads for maximum throughput
    while (end - ptr >= 16) {
        __m256i r0 = _mm256_load_si256((const __m256i*)ptr);
        __m256i r1 = _mm256_load_si256((const __m256i*)(ptr + 8));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);

        __m256i combined = _mm256_or_si256(c0, c1);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(combined));

        if (mask != 0) {
            // Match found in this 16-element block
            int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            if (m0 != 0) return (int)(ptr - data) + __builtin_ctz(m0);
            
            int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            return (int)(ptr - data) + 8 + __builtin_ctz(m1);
        }
        ptr += 16;
    }

    // 3. Middle Tier: Process remaining 8-element chunk
    if (end - ptr >= 8) {
        __m256i r = _mm256_load_si256((const __m256i*)ptr);
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) {
            return (int)(ptr - data) + __builtin_ctz(mask);
        }
        ptr += 8;
    }

    // 4. Scalar Tail
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}