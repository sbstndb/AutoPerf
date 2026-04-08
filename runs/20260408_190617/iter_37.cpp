#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned start to reach 32-byte alignment
    // This ensures the main loop uses aligned loads for maximum throughput
    while (i < size && (reinterpret_cast<std::uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const __m256i target = _mm256_set1_epi32(value);
    
    // 2. Main SIMD Loop - Unrolled by 4 (32 elements per iteration)
    // Processing 128 bytes per loop is optimal for Intel's L1D bandwidth
    const int SIMD_STEP = 8;
    const int UNROLL = 4;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL;

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

        // Combine masks using OR
        __m256i m01 = _mm256_or_si256(c0, c1);
        __m256i m23 = _mm256_or_si256(c2, c3);
        __m256i combined = _mm256_or_si256(m01, m23);

        // Check if any match exists in the block
        if (!_mm256_testz_si256(combined, combined)) {
            int base_idx = static_cast<int>(ptr - data);
            
            int mask;
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c0)))) 
                return base_idx + __builtin_ctz(mask);
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c1)))) 
                return base_idx + 8 + __builtin_ctz(mask);
            if ((mask = _mm256_movemask_ps(_mm256_castsi256_ps(c2)))) 
                return base_idx + 16 + __builtin_ctz(mask);
            
            mask = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
            return base_idx + 24 + __builtin_ctz(mask);
        }
    }

    // 3. Handle remaining elements
    i = static_cast<int>(ptr - data);
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}