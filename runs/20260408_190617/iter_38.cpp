#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Align to 32-byte boundary
    while (i < size && (reinterpret_cast<std::uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    // 2. Main SIMD Loop - Unrolled by 4 for optimal throughput on Intel Ultra
    const int SIMD_STEP = 8;
    const int UNROLL = 4;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL;

    __m256i target = _mm256_set1_epi32(value);

    for (; i <= size - BLOCK_SIZE; i += BLOCK_SIZE) {
        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 24));

        __m256i c0 = _mm256_cmpeq_epi32(v0, target);
        __m256i c1 = _mm256_cmpeq_epi32(v1, target);
        __m256i c2 = _mm256_cmpeq_epi32(v2, target);
        __m256i c3 = _mm256_cmpeq_epi32(v3, target);

        // Combine masks using OR
        __m256i or01 = _mm256_or_si256(c0, c1);
        __m256i or23 = _mm256_or_si256(c2, c3);
        __m256i final_or = _mm256_or_si256(or01, or23);

        // Check if any match exists in the 32-element block
        if (!_mm256_testz_si256(final_or, final_or)) {
            // movemask_ps is faster than movemask_epi8 for 32-bit elements
            int m;
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c0)))) 
                return i + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c1)))) 
                return i + 8 + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c2)))) 
                return i + 16 + __builtin_ctz(m);
            m = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            return i + 24 + __builtin_ctz(m);
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}