#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    float carry = 0.0f;

    // Process in blocks of 8 using AVX2
    if (n >= 8) {
        // Mask for shifting elements within the SIMD register
        // We use permutations to simulate shifts for prefix sum logic
        const __m256i idx_shift1 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
        
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: Shift and add (1-distance)
            // [a, b, c, d, e, f, g, h] + [0, a, b, c, d, e, f, g]
            __m256 t0 = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
            __m256 mask1 = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0));
            x = _mm256_add_ps(x, _mm256_and_ps(t0, mask1));

            // Step 2: Shift and add (2-distance)
            __m256 t1 = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6));
            __m256 mask2 = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, -1, -1, 0, 0));
            x = _mm256_add_ps(x, _mm256_and_ps(t1, mask2));

            // Step 3: Shift and add (4-distance)
            __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20); // Low 128 to High 128
            __m256 mask3 = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
            x = _mm256_add_ps(x, _mm256_and_ps(t2, mask3));

            // Step 4: Add the carry from the previous block
            __m256 v_carry = _mm256_set1_ps(carry);
            x = _mm256_add_ps(x, v_carry);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Extract the last element as the carry for the next block
            // Using a shuffle/extract is faster than memory access
            __m128 high_lane = _mm256_extractf128_ps(x, 1);
            carry = _mm_cvtss_f32(_mm_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3)));
        }
    }

    // Scalar tail handling
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}