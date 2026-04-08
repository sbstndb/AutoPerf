#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned start to reach 32-byte alignment
    while (i < size && (reinterpret_cast<std::uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    // 2. Main SIMD Loop
    // Process 32 elements per iteration (4 * 8-wide AVX2 registers)
    // 4 vectors (128 bytes) is often the sweet spot for AVX2 on modern Intel cores
    const int SIMD_STEP = 8;
    const int UNROLL = 4;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL;

    __m256i target = _mm256_set1_epi32(value);
    const int* ptr = data + i;
    const int* end_ptr = data + size - BLOCK_SIZE;

    for (; ptr <= end_ptr; ptr += BLOCK_SIZE) {
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

        // movemask_ps is faster than movemask_epi8 and sufficient for 32-bit elements
        if (_mm256_movemask_ps(_mm256_castsi256_ps(combined)) != 0) {
            int offset = static_cast<int>(ptr - data);
            int m;
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c0)))) return offset + 0 + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c1)))) return offset + 8 + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c2)))) return offset + 16 + __builtin_ctz(m);
            return offset + 24 + __builtin_ctz(_mm256_movemask_ps(_mm256_castsi256_ps(c3)));
        }
    }

    // 3. Handle remaining elements
    i = static_cast<int>(ptr - data);
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}