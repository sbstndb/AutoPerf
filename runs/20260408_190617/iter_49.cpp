#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    const __m256i target = _mm256_set1_epi32(value);

    // Process 16 elements per iteration (2 x 256-bit vectors)
    // This provides a better balance of ILP and lower branch penalty for this CPU
    while (end - ptr >= 16) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(ptr + 8));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);

        // Combine masks to check 16 elements with one branch
        __m256i combined = _mm256_or_si256(c0, c1);
        int aggregate_mask = _mm256_movemask_ps(_mm256_castsi256_ps(combined));

        if (aggregate_mask != 0) {
            int mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            if (mask0 != 0) {
                return (int)(ptr - data) + __builtin_ctz(mask0);
            }
            int mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            return (int)(ptr - data) + 8 + __builtin_ctz(mask1);
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

    // Scalar tail for the last < 8 elements
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}