#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // Main loop: Process 64 elements per iteration
    if (size >= 64) {
        const int* loop_end = data + (size & ~63);
        for (; ptr < loop_end; ptr += 64) {
            __m256i r0 = _mm256_loadu_si256((const __m256i*)(ptr + 0));
            __m256i r1 = _mm256_loadu_si256((const __m256i*)(ptr + 8));
            __m256i r2 = _mm256_loadu_si256((const __m256i*)(ptr + 16));
            __m256i r3 = _mm256_loadu_si256((const __m256i*)(ptr + 24));
            __m256i r4 = _mm256_loadu_si256((const __m256i*)(ptr + 32));
            __m256i r5 = _mm256_loadu_si256((const __m256i*)(ptr + 40));
            __m256i r6 = _mm256_loadu_si256((const __m256i*)(ptr + 48));
            __m256i r7 = _mm256_loadu_si256((const __m256i*)(ptr + 56));

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
                auto get_index = [&](__m256i mask, const int* p) {
                    int m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
                    return (m != 0) ? (int)(p - data) + __builtin_ctz(m) : -1;
                };

                int res;
                if ((res = get_index(c0, ptr + 0)) != -1) return res;
                if ((res = get_index(c1, ptr + 8)) != -1) return res;
                if ((res = get_index(c2, ptr + 16)) != -1) return res;
                if ((res = get_index(c3, ptr + 24)) != -1) return res;
                if ((res = get_index(c4, ptr + 32)) != -1) return res;
                if ((res = get_index(c5, ptr + 40)) != -1) return res;
                if ((res = get_index(c6, ptr + 48)) != -1) return res;
                return get_index(c7, ptr + 56);
            }
        }
    }

    // Handle 8-element chunks
    while (end - ptr >= 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) return (int)(ptr - data) + __builtin_ctz(mask);
        ptr += 8;
    }

    // Scalar tail
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}