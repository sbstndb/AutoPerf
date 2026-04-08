#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned start to reach 32-byte alignment
    // This allows the use of _mm256_load_si256 (aligned load)
    while (i < size && (reinterpret_cast<std::uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const __m256i target = _mm256_set1_epi32(value);
    const int* ptr = data + i;
    
    // 2. Main SIMD Loop: 4x unrolled (32 elements per iteration)
    // 4x unrolling is the "sweet spot" for Intel Ultra 7 to balance ILP and branch overhead
    const int UNROLL = 4;
    const int SIMD_STEP = 8;
    const int BLOCK_SIZE = UNROLL * SIMD_STEP;

    int remaining = size - i;
    while (remaining >= BLOCK_SIZE) {
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

        if (__builtin_expect(!_mm256_testz_si256(combined, combined), 0)) {
            int m;
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c0)))) return i + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c1)))) return i + 8 + __builtin_ctz(m);
            if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c2)))) return i + 16 + __builtin_ctz(m);
            m = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            return i + 24 + __builtin_ctz(m);
        }

        ptr += BLOCK_SIZE;
        i += BLOCK_SIZE;
        remaining -= BLOCK_SIZE;
    }

    // 3. Handle remaining elements with 1x SIMD steps
    while (remaining >= 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i c = _mm256_cmpeq_epi32(v, target);
        int m = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (m) return i + __builtin_ctz(m);
        
        ptr += 8;
        i += 8;
        remaining -= 8;
    }

    // 4. Final scalar cleanup
    while (i < size) {
        if (data[i] == value) return i;
        i++;
    }

    return -1;
}