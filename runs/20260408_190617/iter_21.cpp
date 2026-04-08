#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Align pointer to 32-byte boundary
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    // 2. Main AVX2 Loop - Unrolled 8x (64 elements per iteration)
    __m256i target = _mm256_set1_epi32(value);
    int vectorized_end = i + ((size - i) & ~63);

    for (; i < vectorized_end; i += 64) {
        // Prefetch data for future iterations
        __builtin_prefetch(&data[i + 256], 0, 3);

        __m256i r0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i]));
        __m256i r1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 8]));
        __m256i r2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 16]));
        __m256i r3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 24]));
        __m256i r4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 32]));
        __m256i r5 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 40]));
        __m256i r6 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 48]));
        __m256i r7 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i + 56]));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);
        __m256i c4 = _mm256_cmpeq_epi32(r4, target);
        __m256i c5 = _mm256_cmpeq_epi32(r5, target);
        __m256i c6 = _mm256_cmpeq_epi32(r6, target);
        __m256i c7 = _mm256_cmpeq_epi32(r7, target);

        // Extract masks (8 bits per movemask_ps)
        uint64_t m0 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c0));
        uint64_t m1 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c1));
        uint64_t m2 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c2));
        uint64_t m3 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c3));
        uint64_t m4 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c4));
        uint64_t m5 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c5));
        uint64_t m6 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c6));
        uint64_t m7 = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(c7));

        // Aggregate all masks into one 64-bit integer
        uint64_t combined = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24) | 
                            (m4 << 32) | (m5 << 40) | (m6 << 48) | (m7 << 56);

        if (combined != 0) {
            return i + (__builtin_ctzll(combined));
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}