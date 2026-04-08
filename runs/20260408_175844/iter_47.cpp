#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x YMM)
    for (; i + 16 <= n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(&input[i]);
        __m256 x1 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane prefix sum for both blocks in parallel
        // Shift and add (1-offset)
        __m256 t0_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x0), 4));
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x0 = _mm256_add_ps(x0, t0_0);
        x1 = _mm256_add_ps(x1, t0_1);

        // Shift and add (2-offset)
        __m256 t1_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x0), 8));
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x0 = _mm256_add_ps(x0, t1_0);
        x1 = _mm256_add_ps(x1, t1_1);

        // Step 2: Cross-lane propagation within each YMM
        // Extract high element of low lane and broadcast to high lane
        __m256 lane_sum0 = _mm256_permute2f128_ps(x0, x0, 0x08);
        lane_sum0 = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(lane_sum0), _MM_SHUFFLE(3, 3, 3, 3)));
        x0 = _mm256_add_ps(x0, lane_sum0);

        __m256 lane_sum1 = _mm256_permute2f128_ps(x1, x1, 0x08);
        lane_sum1 = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(lane_sum1), _MM_SHUFFLE(3, 3, 3, 3)));
        x1 = _mm256_add_ps(x1, lane_sum1);

        // Step 3: Apply global carry to both blocks
        x0 = _mm256_add_ps(x0, v_carry);
        
        // Carry for x1 is (global carry + last element of x0 before global carry was added)
        // Or simply last element of x0 after global carry is added.
        __m256 v_carry_x1 = _mm256_permute2f128_ps(x0, x0, 0x11);
        v_carry_x1 = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(v_carry_x1), _MM_SHUFFLE(3, 3, 3, 3)));
        x1 = _mm256_add_ps(x1, v_carry_x1);

        _mm256_storeu_ps(&output[i], x0);
        _mm256_storeu_ps(&output[i + 8], x1);

        // Update global carry for next iteration
        __m256 v_carry_next = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(v_carry_next), _MM_SHUFFLE(3, 3, 3, 3)));
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 lth = _mm256_permute2f128_ps(x, x, 0x08);
        lth = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(lth), _MM_SHUFFLE(3, 3, 3, 3)));
        x = _mm256_add_ps(x, lth);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        __m256 v_carry_next = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(v_carry_next), _MM_SHUFFLE(3, 3, 3, 3)));
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}