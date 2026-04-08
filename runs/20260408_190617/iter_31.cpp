#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    __m256i target = _mm256_set1_epi32(value);

    // 1. Align to 32-byte boundary for optimal SIMD loading
    // uintptr_t requires <cstdint>
    while (i < size && ((reinterpret_cast<std::uintptr_t>(data + i) & 31) != 0)) {
        if (data[i] == value) return i;
        i++;
    }

    // 2. Main loop: 64 elements per iteration (8 vectors)
    // Using aligned loads for maximum throughput
    for (; i <= size - 64; i += 64) {
        __m256i r0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i r1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        __m256i r2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        __m256i r3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 24));
        __m256i r4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 32));
        __m256i r5 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 40));
        __m256i r6 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 48));
        __m256i r7 = _mm256_load_si256(reinterpret_cast<const __m256i*>(data + i + 56));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);
        __m256i c4 = _mm256_cmpeq_epi32(r4, target);
        __m256i c5 = _mm256_cmpeq_epi32(r5, target);
        __m256i c6 = _mm256_cmpeq_epi32(r6, target);
        __m256i c7 = _mm256_cmpeq_epi32(r7, target);

        __m256i or0123 = _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3));
        __m256i or4567 = _mm256_or_si256(_mm256_or_si256(c4, c5), _mm256_or_si256(c6, c7));
        __m256i combined = _mm256_or_si256(or0123, or4567);

        if (!_mm256_testz_si256(combined, combined)) {
            // Match found: check which specific vector contains the value
            if (!_mm256_testz_si256(or0123, or0123)) {
                int m;
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c0)))) return i + __builtin_ctz(m);
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c1)))) return i + 8 + __builtin_ctz(m);
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c2)))) return i + 16 + __builtin_ctz(m);
                return i + 24 + __builtin_ctz(_mm256_movemask_ps(_mm256_castsi256_ps(c3)));
            } else {
                int m;
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c4)))) return i + 32 + __builtin_ctz(m);
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c5)))) return i + 40 + __builtin_ctz(m);
                if ((m = _mm256_movemask_ps(_mm256_castsi256_ps(c6)))) return i + 48 + __builtin_ctz(m);
                return i + 56 + __builtin_ctz(_mm256_movemask_ps(_mm256_castsi256_ps(c7)));
            }
        }
    }

    // 3. Middle-tier: 8 elements per iteration
    for (; i <= size - 8; i += 8) {
        __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) return i + __builtin_ctz(mask);
    }

    // 4. Final scalar tail
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}