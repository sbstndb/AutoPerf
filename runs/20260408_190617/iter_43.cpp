#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;
    __m256i target = _mm256_set1_epi32(value);

    // Handle small sizes or initial alignment
    if (remaining >= 8) {
        // Align to 32-byte boundary to optimize loads
        // Use size_t for pointer arithmetic to fix compilation error
        size_t ptr_val = reinterpret_cast<size_t>(ptr);
        int alignment = (ptr_val / sizeof(int)) & 7;
        
        if (alignment != 0) {
            int prefix_count = 8 - alignment;
            for (int i = 0; i < prefix_count; ++i) {
                if (ptr[i] == value) return (int)(ptr - data) + i;
            }
            ptr += prefix_count;
            remaining -= prefix_count;
        }

        // Main loop: 4-way unroll (32 elements per iteration)
        while (remaining >= 32) {
            __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 8));
            __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 16));
            __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 24));

            __m256i c0 = _mm256_cmpeq_epi32(v0, target);
            __m256i c1 = _mm256_cmpeq_epi32(v1, target);
            __m256i c2 = _mm256_cmpeq_epi32(v2, target);
            __m256i c3 = _mm256_cmpeq_epi32(v3, target);

            __m256i or01 = _mm256_or_si256(c0, c1);
            __m256i or23 = _mm256_or_si256(c2, c3);
            __m256i combined = _mm256_or_si256(or01, or23);

            if (!_mm256_testz_si256(combined, combined)) {
                int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
                if (m0) return (int)(ptr - data) + __builtin_ctz(m0);
                
                int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
                if (m1) return (int)(ptr - data) + 8 + __builtin_ctz(m1);
                
                int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
                if (m2) return (int)(ptr - data) + 16 + __builtin_ctz(m2);
                
                int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
                return (int)(ptr - data) + 24 + __builtin_ctz(m3);
            }
            ptr += 32;
            remaining -= 32;
        }

        // Handle remaining blocks of 8
        while (remaining >= 8) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i cmp = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask) return (int)(ptr - data) + __builtin_ctz(mask);
            ptr += 8;
            remaining -= 8;
        }
    }

    // Scalar tail for the absolute remainder
    for (int i = 0; i < remaining; ++i) {
        if (ptr[i] == value) return (int)(ptr - data) + i;
    }

    return -1;
}