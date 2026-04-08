#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned prefix to align 'data + i' to 32 bytes
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    __m256i target = _mm256_set1_epi32(value);
    
    // 2. Main AVX2 Vectorized Loop (Unrolled 4x for 32 elements/iter)
    // Processing 128 bytes per iteration to balance ILP and branch prediction
    int vectorized_end = i + ((size - i) & ~31);
    for (; i < vectorized_end; i += 32) {
        // Prefetch data for subsequent iterations
        __builtin_prefetch(&data[i + 64], 0, 3);

        __m256i r0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i]));
        __m256i r1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 8]));
        __m256i r2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 16]));
        __m256i r3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 24]));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);

        __m256i m01 = _mm256_or_si256(c0, c1);
        __m256i m23 = _mm256_or_si256(c2, c3);
        __m256i combined = _mm256_or_si256(m01, m23);

        // Check if any match exists in the 32-element block
        if (_mm256_movemask_ps(_mm256_castsi256_ps(combined)) != 0) {
            int mask;
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c0))) != 0) {
                return i + __builtin_ctz(mask);
            }
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c1))) != 0) {
                return i + 8 + __builtin_ctz(mask);
            }
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c2))) != 0) {
                return i + 16 + __builtin_ctz(mask);
            }
            mask = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            return i + 24 + __builtin_ctz(mask);
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}