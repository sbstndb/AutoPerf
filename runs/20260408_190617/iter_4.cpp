#include <cstddef>
#include <immintrin.h>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;
    // Process 32 elements at a time (4 x 8-wide AVX2 registers)
    const int SIMD_STEP = 8;
    const int UNROLL_FACTOR = 4;
    const int BLOCK_SIZE = SIMD_STEP * UNROLL_FACTOR;

    __m256i target = _mm256_set1_epi32(value);

    for (; i <= size - BLOCK_SIZE; i += BLOCK_SIZE) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 8));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 16));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 24));

        __m256i cmp0 = _mm256_cmpeq_epi32(v0, target);
        __m256i cmp1 = _mm256_cmpeq_epi32(v1, target);
        __m256i cmp2 = _mm256_cmpeq_epi32(v2, target);
        __m256i cmp3 = _mm256_cmpeq_epi32(v3, target);

        // Combine results to check if any match exists in the 32-element block
        __m256i or_combined = _mm256_or_si256(_mm256_or_si256(cmp0, cmp1), _mm256_or_si256(cmp2, cmp3));
        
        if (!_mm256_testz_si256(or_combined, or_combined)) {
            // Match found in this block, find exactly where
            auto check_mask = [&](__m256i cmp, int offset) {
                int mask = _mm256_movemask_epi8(cmp);
                if (mask != 0) {
                    // __builtin_ctz returns trailing zeros. 
                    // Since epi32 uses 4 bytes, we divide by 4 (shift 2)
                    return i + offset + (__builtin_ctz(mask) >> 2);
                }
                return -1;
            };

            int res;
            if ((res = check_mask(cmp0, 0)) != -1) return res;
            if ((res = check_mask(cmp1, 8)) != -1) return res;
            if ((res = check_mask(cmp2, 16)) != -1) return res;
            return check_mask(cmp3, 24);
        }
    }

    // Handle remaining elements
    for (; i < size; i++) {
        if (data[i] == value)
            return i;
    }

    return -1;
}