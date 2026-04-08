#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const int* ptr = data;
    const int* end = data + size;
    __m256i target = _mm256_set1_epi32(value);

    // Process 64-element blocks
    while (ptr <= end - 64) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)(ptr));
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

        int m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0));
        int m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1));
        int m2 = _mm256_movemask_ps(_mm256_castsi256_ps(c2));
        int m3 = _mm256_movemask_ps(_mm256_castsi256_ps(c3));
        int m4 = _mm256_movemask_ps(_mm256_castsi256_ps(c4));
        int m5 = _mm256_movemask_ps(_mm256_castsi256_ps(c5));
        int m6 = _mm256_movemask_ps(_mm256_castsi256_ps(c6));
        int m7 = _mm256_movemask_ps(_mm256_castsi256_ps(c7));

        int combined = m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7;

        if (combined != 0) {
            int offset = 0;
            if (m0) { offset = 0;  goto found; }
            if (m1) { m0 = m1; offset = 8;  goto found; }
            if (m2) { m0 = m2; offset = 16; goto found; }
            if (m3) { m0 = m3; offset = 24; goto found; }
            if (m4) { m0 = m4; offset = 32; goto found; }
            if (m5) { m0 = m5; offset = 40; goto found; }
            if (m6) { m0 = m6; offset = 48; goto found; }
            m0 = m7; offset = 56;
        found:
            int result = (int)(ptr - data) + offset + __builtin_ctz(m0);
            _mm256_zeroupper();
            return result;
        }
        ptr += 64;
    }

    // Process remaining 8-element blocks
    while (ptr <= end - 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)ptr);
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) {
            int result = (int)(ptr - data) + __builtin_ctz(mask);
            _mm256_zeroupper();
            return result;
        }
        ptr += 8;
    }

    _mm256_zeroupper();
    // Final scalar tail
    while (ptr < end) {
        if (*ptr == value) return (int)(ptr - data);
        ptr++;
    }

    return -1;
}