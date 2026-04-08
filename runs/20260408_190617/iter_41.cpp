#include <cstddef>
#include <cstdint>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;

    // 1. Small scalar loop to align ptr to 32-byte boundary
    // This ensures SIMD loads are aligned, avoiding performance penalties.
    while (remaining > 0 && (reinterpret_cast<std::uintptr_t>(ptr) & 31) != 0) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
        remaining--;
    }

    if (remaining >= 8) {
        const __m256i target = _mm256_set1_epi32(value);

        // 2. Main unrolled loop (32 elements per iteration)
        // Unrolling improves throughput by utilizing multiple execution ports.
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

            // movemask_ps is used on casted integer vectors as it has shorter latency than pmovmskb
            if (_mm256_movemask_ps(_mm256_castsi256_ps(combined))) {
                auto get_idx = [&](__m256i cmp, int offset) -> int {
                    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                    if (mask) return (int)(ptr - data) + offset + __builtin_ctz(mask);
                    return -1;
                };

                int res;
                if ((res = get_idx(c0, 0)) != -1) return res;
                if ((res = get_idx(c1, 8)) != -1) return res;
                if ((res = get_idx(c2, 16)) != -1) return res;
                return get_idx(c3, 24);
            }
            ptr += 32;
            remaining -= 32;
        }

        // 3. Single SIMD step (8 elements)
        while (remaining >= 8) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i cmp = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask) {
                return (int)(ptr - data) + __builtin_ctz(mask);
            }
            ptr += 8;
            remaining -= 8;
        }
    }

    // 4. Final scalar tail
    for (int i = 0; i < remaining; i++) {
        if (ptr[i] == value)
            return (int)(ptr - data) + i;
    }

    return -1;
}