#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Align pointer to 32-byte boundary for VMOVDA
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    __m256i target = _mm256_set1_epi32(value);
    
    // 2. Main AVX2 Loop - Unrolled 4x (32 elements / 128 bytes per iter)
    // This fits well within the execution port limits and reduces branch misprediction impact
    int vectorized_end = i + ((size - i) & ~31);
    for (; i < vectorized_end; i += 32) {
        // Prefetch next 2 cache lines (128 bytes ahead)
        _mm_prefetch(reinterpret_cast<const char*>(&data[i + 32]), _MM_HINT_T0);

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
        if (!_mm256_testz_si256(combined, combined)) {
            // Match found, identify which vector and which lane
            auto get_mask = [](__m256i m) { return _mm256_movemask_ps(_mm256_castsi256_ps(m)); };
            
            int mask;
            if ((mask = get_mask(c0)) != 0) return i + __builtin_ctz(mask);
            if ((mask = get_mask(c1)) != 0) return i + 8 + __builtin_ctz(mask);
            if ((mask = get_mask(c2)) != 0) return i + 16 + __builtin_ctz(mask);
            mask = get_mask(c3);
            return i + 24 + __builtin_ctz(mask);
        }
    }

    // 3. Final scalar cleanup
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}