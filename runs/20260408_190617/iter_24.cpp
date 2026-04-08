#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Align pointer to 32-byte boundary for optimized loads
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const __m256i target = _mm256_set1_epi32(value);
    
    // 2. Main Loop: Process 64 elements (256 bytes) per iteration
    int vectorized_end = i + ((size - i) & ~63);
    for (; i < vectorized_end; i += 64) {
        // Prefetch data for future iterations to reduce cache miss stalls
        _mm_prefetch(reinterpret_cast<const char*>(&data[i + 128]), _MM_HINT_T0);

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

        // Extract masks (1 bit per int32)
        int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
        int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
        int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
        int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
        int m4 = _mm256_movemask_ps(_mm256_castsi256_ps(c4));
        int m5 = _mm256_movemask_ps(_mm256_castsi256_ps(c5));
        int m6 = _mm256_movemask_ps(_mm256_castsi256_ps(c6));
        int m7 = _mm256_movemask_ps(_mm256_castsi256_ps(c7));

        // Check if any match in the 64-element block
        if (m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7) {
            if (m0) return i + __builtin_ctz(m0);
            if (m1) return i + 8 + __builtin_ctz(m1);
            if (m2) return i + 16 + __builtin_ctz(m2);
            if (m3) return i + 24 + __builtin_ctz(m3);
            if (m4) return i + 32 + __builtin_ctz(m4);
            if (m5) return i + 40 + __builtin_ctz(m5);
            if (m6) return i + 48 + __builtin_ctz(m6);
            return i + 56 + __builtin_ctz(m7);
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}