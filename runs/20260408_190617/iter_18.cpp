#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;

    if (remaining >= 64) {
        const __m256i target = _mm256_set1_epi32(value);
        
        // Process 64 elements per iteration (8 x 8-way AVX2)
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

            int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
            int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
            int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
            int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            int m4 = _mm256_movemask_ps(_mm256_castsi256_ps(c4));
            int m5 = _mm256_movemask_ps(_mm256_castsi256_ps(c5));
            int m6 = _mm256_movemask_ps(_mm256_castsi256_ps(c6));
            int m7 = _mm256_movemask_ps(_mm256_castsi256_ps(c7));

            int combined_low = m0 | m1 | m2 | m3;
            int combined_high = m4 | m5 | m6 | m7;

            if (combined_low | combined_high) {
                int offset = (int)(ptr - data);
                if (m0) return offset + __builtin_ctz(m0);
                if (m1) return offset + 8 + __builtin_ctz(m1);
                if (m2) return offset + 16 + __builtin_ctz(m2);
                if (m3) return offset + 24 + __builtin_ctz(m3);
                if (m4) return offset + 32 + __builtin_ctz(m4);
                if (m5) return offset + 40 + __builtin_ctz(m5);
                if (m6) return offset + 48 + __builtin_ctz(m6);
                return offset + 56 + __builtin_ctz(m7);
            }

            ptr += 64;
            remaining -= 64;
        }
    }

    // Handle remaining elements in 8-element chunks
    while (remaining >= 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i cmp = _mm256_cmpeq_epi32(v, _mm256_set1_epi32(value));
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        if (mask) {
            return (int)(ptr - data) + __builtin_ctz(mask);
        }
        ptr += 8;
        remaining -= 8;
    }

    // Scalar tail
    for (int i = 0; i < remaining; i++) {
        if (ptr[i] == value)
            return (int)(ptr - data) + i;
    }

    return -1;
}