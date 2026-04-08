#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    int remaining = size;

    if (remaining >= 8) {
        const __m256i target = _mm256_set1_epi32(value);

        // Main Loop: 64 elements per iteration
        while (remaining >= 64) {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 8));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 16));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 24));
            __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 32));
            __m256i v5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 40));
            __m256i v6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 48));
            __m256i v7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 56));

            __m256i c0 = _mm256_cmpeq_epi32(v0, target);
            __m256i c1 = _mm256_cmpeq_epi32(v1, target);
            __m256i c2 = _mm256_cmpeq_epi32(v2, target);
            __m256i c3 = _mm256_cmpeq_epi32(v3, target);
            __m256i c4 = _mm256_cmpeq_epi32(v4, target);
            __m256i c5 = _mm256_cmpeq_epi32(v5, target);
            __m256i c6 = _mm256_cmpeq_epi32(v6, target);
            __m256i c7 = _mm256_cmpeq_epi32(v7, target);

            // Use ORs to create a single mask check
            __m256i m01 = _mm256_or_si256(c0, c1);
            __m256i m23 = _mm256_or_si256(c2, c3);
            __m256i m45 = _mm256_or_si256(c4, c5);
            __m256i m67 = _mm256_or_si256(c6, c7);
            __m256i m03 = _mm256_or_si256(m01, m23);
            __m256i m47 = _mm256_or_si256(m45, m67);
            __m256i combined = _mm256_or_si256(m03, m47);

            // movemask_ps is faster than testz on many Intel cores for this purpose
            if (_mm256_movemask_ps(_mm256_castsi256_ps(combined)) != 0) {
                // Match found, locate exactly where
                auto get_match = [&](__m256i cmp, const int* p) -> int {
                    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
                    return mask ? (int)(p - data) + __builtin_ctz(mask) : -1;
                };

                int res;
                if ((res = get_match(c0, ptr)) != -1) return res;
                if ((res = get_match(c1, ptr + 8)) != -1) return res;
                if ((res = get_match(c2, ptr + 16)) != -1) return res;
                if ((res = get_match(c3, ptr + 24)) != -1) return res;
                if ((res = get_match(c4, ptr + 32)) != -1) return res;
                if ((res = get_match(c5, ptr + 40)) != -1) return res;
                if ((res = get_match(c6, ptr + 48)) != -1) return res;
                return get_match(c7, ptr + 56);
            }

            ptr += 64;
            remaining -= 64;
        }

        // Mid-size tail: 8 elements per iteration
        while (remaining >= 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            __m256i cmp = _mm256_cmpeq_epi32(v, target);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            if (mask) {
                return (int)(ptr - data) + __builtin_ctz(mask);
            }
            ptr += 8;
            remaining -= 8;
        }
    }

    // Final scalar tail
    for (int i = 0; i < remaining; ++i) {
        if (ptr[i] == value) {
            return (int)(ptr - data) + i;
        }
    }

    return -1;
}