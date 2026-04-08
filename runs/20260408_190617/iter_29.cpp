#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    if (size >= 8) {
        __m256i target = _mm256_set1_epi32(value);
        
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

            // Extract masks: movemask_ps is faster than movemask_epi8 for 32-bit lanes
            int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
            int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));

            int combined_mask = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);

            if (combined_mask != 0) {
                // __builtin_ctz returns the index of the first set bit
                // Since we shifted m1, m2, m3, ctz gives the global offset within the 32-block
                return i + __builtin_ctz(combined_mask);
            }
        }

        // Handle remaining 8-element chunks
        for (; i <= size - 8; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            __m256i c = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
            if (mask != 0) {
                return i + __builtin_ctz(mask);
            }
        }
    }

    // Final scalar tail
    for (; i < size; i++) {
        if (data[i] == value)
            return i;
    }

    return -1;
}