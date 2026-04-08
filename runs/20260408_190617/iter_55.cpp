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

    const __m256i target = _mm256_set1_epi32(value);
    const int* ptr = data + i;
    int remaining = size - i;
    
    // 2. Main AVX2 Vectorized Loop (Unrolled 8x for 64 elements/iter)
    while (remaining >= 64) {
        __m256i r0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i r1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 8));
        __m256i r2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 16));
        __m256i r3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 24));
        __m256i r4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 32));
        __m256i r5 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 40));
        __m256i r6 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 48));
        __m256i r7 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 56));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);
        __m256i c4 = _mm256_cmpeq_epi32(r4, target);
        __m256i c5 = _mm256_cmpeq_epi32(r5, target);
        __m256i c6 = _mm256_cmpeq_epi32(r6, target);
        __m256i c7 = _mm256_cmpeq_epi32(r7, target);

        // Extract masks (8 bits per vector)
        uint64_t m0 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c0));
        uint64_t m1 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c1));
        uint64_t m2 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c2));
        uint64_t m3 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c3));
        uint64_t m4 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c4));
        uint64_t m5 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c5));
        uint64_t m6 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c6));
        uint64_t m7 = (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(c7));

        // Combine into one 64-bit mask
        uint64_t combined = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24) | 
                           (m4 << 32) | (m5 << 40) | (m6 << 48) | (m7 << 56);

        if (combined != 0) {
            return (int)(ptr - data) + __builtin_ctzll(combined);
        }

        ptr += 64;
        remaining -= 64;
    }

    // 3. Handle remaining elements
    i = (int)(ptr - data);
    while (i < size) {
        if (data[i] == value) return i;
        i++;
    }

    return -1;
}