#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane scan for both blocks in parallel
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        x2 = _mm256_add_ps(x2, t0_2);

        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t1_2);

        // Step 2: Cross-lane propagation for x1
        __m256 x1_low = _mm256_permute2f128_ps(x1, x1, 0x08);
        x1_low = _mm256_permute_ps(x1_low, 0xFF);
        x1 = _mm256_add_ps(x1, x1_low);
        
        // Apply global carry to x1
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Step 3: Extract carry from x1 for x2
        __m128 x1_high = _mm256_extractf128_ps(x1, 1);
        __m128 v_carry_x2_128 = _mm_shuffle_ps(x1_high, x1_high, _MM_SHUFFLE(3, 3, 3, 3));
        __m256 v_carry_x2 = _mm256_set_m128(v_carry_x2_128, v_carry_x2_128);

        // Step 4: Cross-lane propagation for x2 and apply carry
        __m256 x2_low = _mm256_permute2f128_ps(x2, x2, 0x08);
        x2_low = _mm256_permute_ps(x2_low, 0xFF);
        x2 = _mm256_add_ps(x2, x2_low);
        x2 = _mm256_add_ps(x2, v_carry_x2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Step 5: Update global carry for next iteration
        __m128 x2_high = _mm256_extractf128_ps(x2, 1);
        __m128 v_carry_next_128 = _mm_shuffle_ps(x2_high, x2_high, _MM_SHUFFLE(3, 3, 3, 3));
        v_carry = _mm256_set_m128(v_carry_next_128, v_carry_next_128);
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 low_to_high = _mm256_permute2f128_ps(x, x, 0x08);
        low_to_high = _mm256_permute_ps(low_to_high, 0xFF);
        x = _mm256_add_ps(x, low_to_high);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        __m128 x_high = _mm256_extractf128_ps(x, 1);
        __m128 next_c = _mm_shuffle_ps(x_high, x_high, _MM_SHUFFLE(3, 3, 3, 3));
        v_carry = _mm256_set_m128(next_c, next_c);
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}