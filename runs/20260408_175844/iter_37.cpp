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
        // Shift 4 bytes (1 float)
        __m256 t1_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t2_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t1_0);
        x2 = _mm256_add_ps(x2, t2_0);

        // Shift 8 bytes (2 floats)
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t2_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t2_1);

        // Step 2: Cross-lane propagation within each YMM
        // Broadcast the sum of the lower 128-bit lane to the upper 128-bit lane
        __m256 x1_low = _mm256_permute2f128_ps(x1, x1, 0x08);
        __m256 x2_low = _mm256_permute2f128_ps(x2, x2, 0x08);
        x1_low = _mm256_permute_ps(x1_low, 0xFF);
        x2_low = _mm256_permute_ps(x2_low, 0xFF);
        x1 = _mm256_add_ps(x1, x1_low);
        x2 = _mm256_add_ps(x2, x2_low);

        // Step 3: Apply carry from previous 16-element iteration to x1
        x1 = _mm256_add_ps(x1, v_carry);

        // Step 4: Propagate x1's last element to all of x2
        __m256 x1_last = _mm256_permute2f128_ps(x1, x1, 0x11);
        x1_last = _mm256_permute_ps(x1_last, 0xFF);
        x2 = _mm256_add_ps(x2, x1_last);

        _mm256_storeu_ps(&output[i], x1);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Prepare carry for next iteration
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
        __m256 low_to_high = _mm256_permute2f128_ps(x, x, 0x08);
        low_to_high = _mm256_permute_ps(low_to_high, 0xFF);
        x = _mm256_add_ps(x, low_to_high);
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