#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const __m256i target = _mm256_set1_epi32(value);
    int i = 0;

    // Process 64 elements at a time (8 x 8-wide AVX2 registers)
    // This maximizes ILP on Intel Ultra architectures
    const int UNROLL_FACTOR = 8;
    const int SIMD_STEP = 8;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL_FACTOR;

    for (; i <= size - BLOCK_SIZE; i += BLOCK_SIZE) {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        const __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 24));
        const __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 32));
        const __m256i v5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 40));
        const __m256i v6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 48));
        const __m256i v7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 56));

        const __m256i c0 = _mm256_cmpeq_epi32(v0, target);
        const __m256i c1 = _mm256_cmpeq_epi32(v1, target);
        const __m256i c2 = _mm256_cmpeq_epi32(v2, target);
        const __m256i c3 = _mm256_cmpeq_epi32(v3, target);
        const __m256i c4 = _mm256_cmpeq_epi32(v4, target);
        const __m256i c5 = _mm256_cmpeq_epi32(v5, target);
        const __m256i c6 = _mm256_cmpeq_epi32(v6, target);
        const __m256i c7 = _mm256_cmpeq_epi32(v7, target);

        const __m256i or01 = _mm256_or_si256(c0, c1);
        const __m256i or23 = _mm256_or_si256(c2, c3);
        const __m256i or45 = _mm256_or_si256(c4, c5);
        const __m256i or67 = _mm256_or_si256(c6, c7);
        
        const __m256i combined = _mm256_or_si256(_mm256_or_si256(or01, or23), 
                                                 _mm256_or_si256(or45, or67));

        if (!_mm256_testz_si256(combined, combined)) {
            auto get_index = [&](__m256i cmp) -> int {
                // movemask_ps is faster and gives 1 bit per 32-bit lane
                int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                return (mask != 0) ? __builtin_ctz(mask) : -1;
            };

            int idx;
            if ((idx = get_index(c0)) != -1) return i + idx;
            if ((idx = get_index(c1)) != -1) return i + 8 + idx;
            if ((idx = get_index(c2)) != -1) return i + 16 + idx;
            if ((idx = get_index(c3)) != -1) return i + 24 + idx;
            if ((idx = get_index(c4)) != -1) return i + 32 + idx;
            if ((idx = get_index(c5)) != -1) return i + 40 + idx;
            if ((idx = get_index(c6)) != -1) return i + 48 + idx;
            return i + 56 + get_index(c7);
        }
    }

    // Scalar tail
    for (; i < size; ++i) {
        if (data[i] == value) return i;
    }

    return -1;
}