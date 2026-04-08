#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    __m256i target = _mm256_set1_epi32(value);

    // Process in blocks of 32 (4 * 8)
    // We use movemask_ps on the result of epi32 comparisons.
    // This is valid because a match results in all 1s (NaN in float), 
    // and movemask_ps extracts the MSB of each 32-bit lane.
    for (; i <= size - 32; i += 32) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 24));

        __m256i c0 = _mm256_cmpeq_epi32(v0, target);
        __m256i c1 = _mm256_cmpeq_epi32(v1, target);
        __m256i c2 = _mm256_cmpeq_epi32(v2, target);
        __m256i c3 = _mm256_cmpeq_epi32(v3, target);

        // Combine using OR to check 32 elements with one branch
        __m256i m01 = _mm256_or_si256(c0, c1);
        __m256i m23 = _mm256_or_si256(c2, c3);
        __m256i combined = _mm256_or_si256(m01, m23);

        if (!_mm256_testz_si256(combined, combined)) {
            // Extract masks using movemask_ps (1 bit per 32-bit int)
            int mask0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            if (mask0 != 0) return i + __builtin_ctz(mask0);
            
            int mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            if (mask1 != 0) return i + 8 + __builtin_ctz(mask1);
            
            int mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
            if (mask2 != 0) return i + 16 + __builtin_ctz(mask2);
            
            int mask3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            return i + 24 + __builtin_ctz(mask3);
        }
    }

    // Handle remaining elements with a smaller SIMD step to minimize scalar overhead
    for (; i <= size - 8; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i c = _mm256_cmpeq_epi32(v, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) return i + __builtin_ctz(mask);
    }

    // Final scalar tail
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}