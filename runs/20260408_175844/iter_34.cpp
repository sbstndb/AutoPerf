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

        // Block 1: Intra-lane scan
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t0);
        t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t0);

        // Block 1: Cross-lane
        __m256 lane_sum1 = _mm256_permute2f128_ps(x1, x1, 0x08); // Low 128 to high 128
        lane_sum1 = _mm256_shuffle_ps(lane_sum1, lane_sum1, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, lane_sum1);
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Extract carry for Block 2 from the last element of Block 1
        __m128 x1_high = _mm256_extractf128_ps(x1, 1);
        __m256 v_carry2 = _mm256_broadcast_ss((float*)&x1_high + 3);

        // Block 2: Intra-lane scan
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x2 = _mm256_add_ps(x2, t1);
        t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x2 = _mm256_add_ps(x2, t1);

        // Block 2: Cross-lane
        __m256 lane_sum2 = _mm256_permute2f128_ps(x2, x2, 0x08);
        lane_sum2 = _mm256_shuffle_ps(lane_sum2, lane_sum2, _MM_SHUFFLE(3, 3, 3, 3));
        x2 = _mm256_add_ps(x2, lane_sum2);
        x2 = _mm256_add_ps(x2, v_carry2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Update carry for next 16-element iteration
        __m128 x2_high = _mm256_extractf128_ps(x2, 1);
        v_carry = _mm256_broadcast_ss((float*)&x2_high + 3);
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t);
        t = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t);
        
        __m256 lane_sum = _mm256_permute2f128_ps(x, x, 0x08);
        lane_sum = _mm256_shuffle_ps(lane_sum, lane_sum, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, lane_sum);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        __m128 x_high = _mm256_extractf128_ps(x, 1);
        v_carry = _mm256_broadcast_ss((float*)&x_high + 3);
    }

    // Scalar tail
    float carry_val;
    if (i > 0) {
        carry_val = output[i - 1];
    } else {
        carry_val = 0.0f;
    }

    for (; i < n; ++i) {
        carry_val += input[i];
        output[i] = carry_val;
    }
}