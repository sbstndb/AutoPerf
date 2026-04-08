#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    // Broadcast the search value to all 8 lanes of a 256-bit register
    __m256i target = _mm256_set1_epi32(value);

    // Unroll loop to process 32 elements (4 x 8) per iteration
    for (; i <= size - 32; i += 32) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(data + i + 8));
        __m256i r2 = _mm256_loadu_si256((const __m256i*)(data + i + 16));
        __m256i r3 = _mm256_loadu_si256((const __m256i*)(data + i + 24));

        __m256i c0 = _mm256_cmpeq_epi32(r0, target);
        __m256i c1 = _mm256_cmpeq_epi32(r1, target);
        __m256i c2 = _mm256_cmpeq_epi32(r2, target);
        __m256i c3 = _mm256_cmpeq_epi32(r3, target);

        // Combine masks to check if any match exists in this block of 32
        __m256i or_block = _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3));
        if (!_mm256_testz_si256(or_block, or_block)) {
            // Match found in this block, find exactly which one
            auto get_index = [&](__m256i mask, int offset) {
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
                if (m != 0) return offset + __builtin_ctz(m);
                return -1;
            };

            int res;
            if ((res = get_index(c0, i)) != -1) return res;
            if ((res = get_index(c1, i + 8)) != -1) return res;
            if ((res = get_index(c2, i + 16)) != -1) return res;
            return get_index(c3, i + 24);
        }
    }

    // Process remaining 8-element chunks
    for (; i <= size - 8; i += 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i c = _mm256_cmpeq_epi32(r, target);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(c));
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }

    // Scalar tail for remaining elements (< 8)
    for (; i < size; i++) {
        if (data[i] == value)
            return i;
    }

    return -1;
}