#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time to hide latency
    for (; i + 16 <= n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane prefix sum for both blocks (Parallel)
        // Shift and add: [a, b, c, d] -> [a, a+b, b+c, c+d]
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        x2 = _mm256_add_ps(x2, t0_2);

        // Shift and add: [a, b, c, d] -> [a, b, a+c, b+d]
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t1_2);

        // Step 2: Cross-lane for Block 1
        // Broadcast the high element of the low lane to the high lane
        __m256 lane_carry1 = _mm256_permute2f128_ps(x1, x1, 0x00); // [L, L]
        lane_carry1 = _mm256_permute_ps(lane_carry1, _MM_SHUFFLE(3, 3, 3, 3));
        // Mask out the low lane so only high lane gets the carry
        lane_carry1 = _mm256_blend_ps(_mm256_setzero_ps(), lane_carry1, 0xF0);
        x1 = _mm256_add_ps(x1, lane_carry1);
        
        // Apply carry from previous 16-element iteration
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Step 3: Cross-lane for Block 2
        __m256 lane_carry2 = _mm256_permute2f128_ps(x2, x2, 0x00);
        lane_carry2 = _mm256_permute_ps(lane_carry2, _MM_SHUFFLE(3, 3, 3, 3));
        lane_carry2 = _mm256_blend_ps(_mm256_setzero_ps(), lane_carry2, 0xF0);
        x2 = _mm256_add_ps(x2, lane_carry2);

        // Carry for Block 2 is the last element of Block 1
        __m256 v_carry_mid = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry_mid = _mm256_permute_ps(v_carry_mid, _MM_SHUFFLE(3, 3, 3, 3));
        
        x2 = _mm256_add_ps(x2, v_carry_mid);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Update global carry for next loop
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Handle remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        
        __m256 lc = _mm256_permute2f128_ps(x, x, 0x00);
        lc = _mm256_permute_ps(lc, _MM_SHUFFLE(3, 3, 3, 3));
        lc = _mm256_blend_ps(_mm256_setzero_ps(), lc, 0xF0);
        x = _mm256_add_ps(x, lc);
        x = _mm256_add_ps(x, v_carry);
        
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}