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

    const __m256i target = _mm256_set1_epi32(value);
    const int* ptr = data + i;
    int remaining = size - i;

    // 2. Main Loop: Process 64 elements (256 bytes) per iteration
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

        __m256i m01 = _mm256_or_si256(c0, c1);
        __m256i m23 = _mm256_or_si256(c2, c3);
        __m256i m45 = _mm256_or_si256(c4, c5);
        __m256i m67 = _mm256_or_si256(c6, c7);
        __m256i m03 = _mm256_or_si256(m01, m23);
        __m256i m47 = _mm256_or_si256(m45, m67);
        __m256i combined = _mm256_or_si256(m03, m47);

        if (!_mm256_testz_si256(combined, combined)) {
            // Match found, identify which vector and which lane
            auto get_idx = [&](__m256i mask) {
                return _mm256_movemask_ps(_mm256_castsi256_ps(mask));
            };

            int mask;
            if ((mask = get_idx(c0))) return (int)(ptr - data) + __builtin_ctz(mask);
            if ((mask = get_idx(c1))) return (int)(ptr - data) + 8 + __builtin_ctz(mask);
            if ((mask = get_idx(c2))) return (int)(ptr - data) + 16 + __builtin_ctz(mask);
            if ((mask = get_idx(c3))) return (int)(ptr - data) + 24 + __builtin_ctz(mask);
            if ((mask = get_idx(c4))) return (int)(ptr - data) + 32 + __builtin_ctz(mask);
            if ((mask = get_idx(c5))) return (int)(ptr - data) + 40 + __builtin_ctz(mask);
            if ((mask = get_idx(c6))) return (int)(ptr - data) + 48 + __builtin_ctz(mask);
            mask = get_idx(c7);
            return (int)(ptr - data) + 56 + __builtin_ctz(mask);
        }

        ptr += 64;
        remaining -= 64;
    }

    // 3. Cleanup
    i = size - remaining;
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}