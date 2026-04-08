#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 8 elements at a time using AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: Intra-lane scan (0-3 and 4-7 independently)
        // Shift left by 1 float (4 bytes)
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        // Shift left by 2 floats (8 bytes)
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);

        // Step 2: Inter-lane propagation (propagate sum of index 3 to indices 4-7)
        // Grab the low 128-bit lane, broadcast the 4th element (index 3) to all positions
        __m256 low_lane_sum = _mm256_permute2f128_ps(x, x, 0x00); 
        __m256 carry_from_low = _mm256_permute_ps(low_lane_sum, 0xFF); // Broadcast index 3
        
        // Zero out the low lane of the carry so we only add to the high lane
        __m256 mask = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
        carry_from_low = _mm256_and_ps(carry_from_low, mask);
        x = _mm256_add_ps(x, carry_from_low);

        // Step 3: Add carry from previous 8-element blocks
        x = _mm256_add_ps(x, v_carry);

        _mm256_storeu_ps(&output[i], x);

        // Step 4: Prepare carry for next iteration (broadcast index 7)
        __m256 high_lane_sum = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(high_lane_sum, 0xFF);
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}