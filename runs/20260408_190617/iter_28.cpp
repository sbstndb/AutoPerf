#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    const __m256i target = _mm256_set1_epi32(value);

    // SIMD Path: AVX2
    if (size >= 8) {
        // Process 32 elements at a time
        for (; i <= size - 32; i += 32) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 16));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 24));

            __m256i c0 = _mm256_cmpeq_epi32(v0, target);
            __m256i c1 = _mm256_cmpeq_epi32(v1, target);
            __m256i c2 = _mm256_cmpeq_epi32(v2, target);
            __m256i c3 = _mm256_cmpeq_epi32(v3, target);

            // Combine all comparisons
            __m256i or01 = _mm256_or_si256(c0, c1);
            __m256i or23 = _mm256_or_si256(c2, c3);
            __m256i combined = _mm256_or_si256(or01, or23);

            // movemask_ps is faster than movemask_epi8 for 32-bit lanes
            if (_mm256_movemask_ps(_mm256_castsi256_ps(combined)) != 0) {
                // Match found in this block of 32
                int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
                if (m0) return i + __builtin_ctz(m0);
                
                int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
                if (m1) return i + 8 + __builtin_ctz(m1);
                
                int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
                if (m2) return i + 16 + __builtin_ctz(m2);
                
                int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
                return i + 24 + __builtin_ctz(m3);
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