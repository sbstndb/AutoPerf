#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2 x YMM)
    // We process x1 and x2 in parallel to maximize ILP
    for (; i + 15 < n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane scan for both blocks
        // Shift and add 4 bytes (1 float)
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        x2 = _mm256_add_ps(x2, t0_2);

        // Shift and add 8 bytes (2 floats)
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t1_2);

        // Step 2: Inter-lane scan (cross 128-bit boundary)
        // x1: Add sum of low 4 floats to high 4 floats
        __m256 lane_sum1 = _mm256_permute2f128_ps(x1, x1, 0x20); 
        x1 = _mm256_add_ps(x1, _mm256_permute_ps(lane_sum1, 0xFF));
        
        // x2: Add sum of low 4 floats to high 4 floats
        __m256 lane_sum2 = _mm256_permute2f128_ps(x2, x2, 0x20);
        x2 = _mm256_add_ps(x2, _mm256_permute_ps(lane_sum2, 0xFF));

        // Step 3: Apply carries
        // Apply global carry to x1
        x1 = _mm256_add_ps(x1, v_carry);
        
        // Carry for x2 is the last element of the finished x1
        __m256 v_carry_x2 = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry_x2 = _mm256_permute_ps(v_carry_x2, 0xFF);
        x2 = _mm256_add_ps(x2, v_carry_x2);

        _mm256_storeu_ps(&output[i], x1);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Update global carry for next 16-element iteration
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Process remaining 8-element blocks
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20);
        x = _mm256_add_ps(x, _mm256_permute_ps(t2, 0xFF));
        
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}