#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x unrolling)
    for (; i + 16 <= n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane scan for both blocks
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        x2 = _mm256_add_ps(x2, t0_2);

        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t1_2);

        // Step 2: Cross-lane scan for x1
        __m256 lane_sum1 = _mm256_permute2f128_ps(x1, x1, 0x08); 
        lane_sum1 = _mm256_permute_ps(lane_sum1, 0xFF);
        x1 = _mm256_add_ps(x1, lane_sum1);
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Step 3: Calculate carry for x2
        // Instead of waiting for x1's full result, we extract the last element of x1
        __m256 v_carry2 = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry2 = _mm256_permute_ps(v_carry2, 0xFF);

        // Step 4: Cross-lane scan for x2
        __m256 lane_sum2 = _mm256_permute2f128_ps(x2, x2, 0x08);
        lane_sum2 = _mm256_permute_ps(lane_sum2, 0xFF);
        x2 = _mm256_add_ps(x2, lane_sum2);
        x2 = _mm256_add_ps(x2, v_carry2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Step 5: Update carry for next iteration
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        
        __m256 lth = _mm256_permute2f128_ps(x, x, 0x08);
        lth = _mm256_permute_ps(lth, 0xFF);
        x = _mm256_add_ps(x, lth);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}