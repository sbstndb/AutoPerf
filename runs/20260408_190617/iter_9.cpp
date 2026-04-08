#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // Process 64 elements at a time (8 x 8-wide AVX2 registers)
    // This unrolling factor balances register pressure and ILP on Intel Ultra architectures
    while (end - ptr >= 64) {
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

        __m256i or01 = _mm256_or_si256(c0, c1);
        __m256i or23 = _mm256_or_si256(c2, c3);
        __m256i or45 = _mm256_or_si256(c4, c5);
        __m256i or67 = _mm256_or_si256(c6, c7);

        __m256i or_all = _mm256_or_si256(_mm256_or_si256(or01, or23), _mm256_or_si256(or45, or67));

        if (!_mm256_testz_si256(or_all, or_all)) {
            // Find which specific register contains the match
            auto get_idx = [&](__m256i cmp, const int* base_ptr) {
                int mask = _mm256_movemask_epi8(cmp);
                if (mask) return (int)(base_ptr - data) + (__builtin_ctz(mask) >> 2);
                return -1;
            };

            int res;
            if ((res = get_idx(c0, ptr)) != -1) return res;
            if ((res = get_idx(c1, ptr + 8)) != -1) return res;
            if ((res = get_idx(c2, ptr + 16)) != -1) return res;
            if ((res = get_idx(c3, ptr + 24)) != -1) return res;
            if ((res = get_idx(c4, ptr + 32)) != -1) return res;
            if ((res = get_idx(c5, ptr + 40)) != -1) return res;
            if ((res = get_idx(c6, ptr + 48)) != -1) return res;
            return get_idx(c7, ptr + 56);
        }
        ptr += 64;
    }

    // Handle remaining elements with 8-wide SIMD
    while (end - ptr >= 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i cmp = _mm256_cmpeq_epi32(v, target);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask) {
            return (int)(ptr - data) + (__builtin_ctz(mask) >> 2);
        }
        ptr += 8;
    }

    // Scalar tail
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}