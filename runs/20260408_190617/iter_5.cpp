#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned prefix to align 'data + i' to 32 bytes
    // This improves performance for large arrays by ensuring aligned loads in the loop.
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    // 2. Main AVX2 Vectorized Loop (Unrolled 4x)
    __m256i target = _mm256_set1_epi32(value);
    int vectorized_end = i + ((size - i) & ~31);

    for (; i < vectorized_end; i += 32) {
        __m256i r0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i]));
        __m256i r1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 8]));
        __m256i r2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 16]));
        __m256i r3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 24]));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);

        // Combine results to check if any match exists in these 32 elements
        __m256i or_01 = _mm256_or_si256(c0, c1);
        __m256i or_23 = _mm256_or_si256(c2, c3);
        __m256i combined = _mm256_or_si256(or_01, or_23);

        if (!_mm256_testz_si256(combined, combined)) {
            // Match found in this block, find exactly where
            auto get_index = [&](__m256i mask, int offset) -> int {
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
                if (m != 0) return offset + __builtin_ctz(m);
                return -1;
            };

            int res;
            if ((res = get_index(c0, i)) != -1) return res;
            if ((res = get_index(c1, i + 8)) != -1) return res;
            if ((res = get_index(c2, i + 16)) != -1) return res;
            return get_index(c3, i + 24);
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}