#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Scalar preamble to align pointer to 32-byte boundary for optimal SIMD loads
    while (i < size && (reinterpret_cast<std::uintptr_t>(data + i) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const int SIMD_STEP = 8;
    const int UNROLL_FACTOR = 8;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL_FACTOR; // 64 elements

    __m256i target = _mm256_set1_epi32(value);

    // 2. Main Loop: Process 64 elements per iteration
    for (; i <= size - BLOCK_SIZE; i += BLOCK_SIZE) {
        // Prefetch data for future iterations
        _mm_prefetch(reinterpret_cast<const char*>(data + i + BLOCK_SIZE * 2), _MM_HINT_T0);

        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 24));
        __m256i v4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 32));
        __m256i v5 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 40));
        __m256i v6 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 48));
        __m256i v7 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 56));

        __m256i c0 = _mm256_cmpeq_epi32(v0, target);
        __m256i c1 = _mm256_cmpeq_epi32(v1, target);
        __m256i c2 = _mm256_cmpeq_epi32(v2, target);
        __m256i c3 = _mm256_cmpeq_epi32(v3, target);
        __m256i c4 = _mm256_cmpeq_epi32(v4, target);
        __m256i c5 = _mm256_cmpeq_epi32(v5, target);
        __m256i c6 = _mm256_cmpeq_epi32(v6, target);
        __m256i c7 = _mm256_cmpeq_epi32(v7, target);

        // Aggregate results using ORs
        __m256i or01 = _mm256_or_si256(c0, c1);
        __m256i or23 = _mm256_or_si256(c2, c3);
        __m256i or45 = _mm256_or_si256(c4, c5);
        __m256i or67 = _mm256_or_si256(c6, c7);
        __m256i combined = _mm256_or_si256(_mm256_or_si256(or01, or23), _mm256_or_si256(or45, or67));

        // Check if any match exists in the 64-element block
        if (!_mm256_testz_si256(combined, combined)) {
            auto get_idx = [](const __m256i& cmp) { 
                return _mm256_movemask_ps(_mm256_castsi256_ps(cmp)); 
            };
            
            int m;
            if ((m = get_idx(c0))) return i + 0 + __builtin_ctz(m);
            if ((m = get_idx(c1))) return i + 8 + __builtin_ctz(m);
            if ((m = get_idx(c2))) return i + 16 + __builtin_ctz(m);
            if ((m = get_idx(c3))) return i + 24 + __builtin_ctz(m);
            if ((m = get_idx(c4))) return i + 32 + __builtin_ctz(m);
            if ((m = get_idx(c5))) return i + 40 + __builtin_ctz(m);
            if ((m = get_idx(c6))) return i + 48 + __builtin_ctz(m);
            return i + 56 + __builtin_ctz(get_idx(c7));
        }
    }

    // 3. Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}