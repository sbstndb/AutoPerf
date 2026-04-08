#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x YMM)
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
        __m256 x1_hi = _mm256_permute2f128_ps(x1, x1, 0x08); // Low lane to high lane
        x1_hi = _mm256_permute_ps(x1_hi, 0xFF);             // Broadcast last element of low lane
        x1 = _mm256_add_ps(x1, x1_hi);
        x1 = _mm256_add_ps(x1, v_carry);                    // Add carry from previous 16-block
        _mm256_storeu_ps(&output[i], x1);

        // Step 3: Propagate carry from x1 to x2
        // Extract the last element of x1 (index 7) to use as carry for x2
        __m128 x1_end = _mm256_extractf128_ps(x1, 1);
        __m256 v_carry_x2 = _mm256_broadcast_ss((float*)&x1_end + 3);

        // Step 4: Cross-lane propagation for x2
        __m256 x2_hi = _mm256_permute2f128_ps(x2, x2, 0x08);
        x2_hi = _mm256_permute_ps(x2_hi, 0xFF);
        x2 = _mm256_add_ps(x2, x2_hi);
        x2 = _mm256_add_ps(x2, v_carry_x2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Step 5: Prepare carry for next iteration
        __m128 x2_end = _mm256_extractf128_ps(x2, 1);
        v_carry = _mm256_broadcast_ss((float*)&x2_end + 3);
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 hi = _mm256_permute2f128_ps(x, x, 0x08);
        hi = _mm256_permute_ps(hi, 0xFF);
        x = _mm256_add_ps(x, hi);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        __m128 x_end = _mm256_extractf128_ps(x, 1);
        v_carry = _mm256_broadcast_ss((float*)&x_end + 3);
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}