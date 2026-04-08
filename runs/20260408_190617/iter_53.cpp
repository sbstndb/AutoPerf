#include <immintrin.h>
#include <cstddef>
#include <cstdint>

int find(const int* data, int size, int value) {
    if (size <= 0) return -1;

    int i = 0;

    // 1. Handle unaligned prefix to align 'data + i' to 32 bytes
    while (i < size && (reinterpret_cast<uintptr_t>(&data[i]) & 31) != 0) {
        if (data[i] == value) return i;
        i++;
    }

    const __m256i target = _mm256_set1_epi32(value);
    const int* ptr = data + i;
    
    // 2. Main AVX2 Vectorized Loop (Unrolled 8x for 64 elements/iter)
    int remaining_elements = size - i;
    int vectorized_count = remaining_elements & ~63;
    const int* end_ptr = ptr + vectorized_count;

    for (; ptr < end_ptr; ptr += 64) {
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
            auto get_index = [&](__m256i mask, const int* base_ptr) -> int {
                int m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
                if (m == 0) return -1;
                return static_cast<int>(base_ptr - data) + __builtin_ctz(m);
            };

            int res;
            if ((res = get_index(c0, ptr)) != -1) return res;
            if ((res = get_index(c1, ptr + 8)) != -1) return res;
            if ((res = get_index(c2, ptr + 16)) != -1) return res;
            if ((res = get_index(c3, ptr + 24)) != -1) return res;
            if ((res = get_index(c4, ptr + 32)) != -1) return res;
            if ((res = get_index(c5, ptr + 40)) != -1) return res;
            if ((res = get_index(c6, ptr + 48)) != -1) return res;
            return get_index(c7, ptr + 56);
        }
    }

    // 3. Handle remaining elements
    i = static_cast<int>(ptr - data);
    while (i < size) {
        if (data[i] == value) return i;
        i++;
    }

    return -1;
}