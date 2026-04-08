#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    const __m256i target = _mm256_set1_epi32(value);
    int i = 0;

    // Unroll to 64 elements (8 * 8) to maximize ILP and saturate load ports
    for (; i <= size - 64; i += 64) {
        __m256i r0 = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i r1 = _mm256_loadu_si256((const __m256i*)(data + i + 8));
        __m256i r2 = _mm256_loadu_si256((const __m256i*)(data + i + 16));
        __m256i r3 = _mm256_loadu_si256((const __m256i*)(data + i + 24));
        __m256i r4 = _mm256_loadu_si256((const __m256i*)(data + i + 32));
        __m256i r5 = _mm256_loadu_si256((const __m256i*)(data + i + 40));
        __m256i r6 = _mm256_loadu_si256((const __m256i*)(data + i + 48));
        __m256i r7 = _mm256_loadu_si256((const __m256i*)(data + i + 56));

        __m256i c01 = _mm256_or_si256(_mm256_cmpeq_epi32(r0, target), _mm256_cmpeq_epi32(r1, target));
        __m256i c23 = _mm256_or_si256(_mm256_cmpeq_epi32(r2, target), _mm256_cmpeq_epi32(r3, target));
        __m256i c45 = _mm256_or_si256(_mm256_cmpeq_epi32(r4, target), _mm256_cmpeq_epi32(r5, target));
        __m256i c67 = _mm256_or_si256(_mm256_cmpeq_epi32(r6, target), _mm256_cmpeq_epi32(r7, target));

        __m256i combined = _mm256_or_si256(_mm256_or_si256(c01, c23), _mm256_or_si256(c45, c67));

        // Single branch for the entire 64-element block
        if (!_mm256_testz_si256(combined, combined)) {
            auto check_lane = [&](int idx) {
                __m256i r = _mm256_loadu_si256((const __m256i*)(data + idx));
                int mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(r, target)));
                if (mask != 0) return idx + __builtin_ctz(mask);
                return -1;
            };

            for (int j = 0; j < 64; j += 8) {
                int res = check_lane(i + j);
                if (res != -1) return res;
            }
        }
    }

    // Handle remaining 8-element chunks
    for (; i <= size - 8; i += 8) {
        __m256i r = _mm256_loadu_si256((const __m256i*)(data + i));
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(r, target)));
        if (mask != 0) return i + __builtin_ctz(mask);
    }

    // Scalar tail
    for (; i < size; i++) {
        if (data[i] == value) return i;
    }

    return -1;
}