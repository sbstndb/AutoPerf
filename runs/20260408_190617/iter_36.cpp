#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned start to reach 32-byte alignment
    // This ensures vmovdqa (aligned loads) can be used or that vmovdqu doesn't cross cache lines
    while (i < size && (reinterpret_cast<std::uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const int* ptr = data + i;
    int remaining = size - i;
    __m256i target = _mm256_set1_epi32(value);

    // 2. Main SIMD Loop (Unrolled by 4 for better ILP and AGU efficiency)
    while (remaining >= 32) {
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

        if (!_mm256_testz_si256(combined, combined)) {
            // Found a match, identify which vector
            auto get_pos = []( __m256i c) -> int {
                int mask = _mm256_movemask_epi8(c);
                return __builtin_ctz(mask) >> 2; // divide by 4 because epi8 mask has 4 bits per int32
            };

            int m;
            if ((m = _mm256_movemask_epi8(c0))) return (int)(ptr - data) + (__builtin_ctz(m) >> 2);
            if ((m = _mm256_movemask_epi8(c1))) return (int)(ptr - data) + 8 + (__builtin_ctz(m) >> 2);
            if ((m = _mm256_movemask_epi8(c2))) return (int)(ptr - data) + 16 + (__builtin_ctz(m) >> 2);
            return (int)(ptr - data) + 24 + get_pos(c3);
        }

        ptr += 32;
        remaining -= 32;
        
        // Prefetch next block to hide memory latency
        _mm_prefetch(reinterpret_cast<const char*>(ptr + 64), _MM_HINT_T0);
    }

    // 3. Handle remaining elements
    i = (int)(ptr - data);
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}