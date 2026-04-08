#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x YMM) to hide latency
    for (; i + 16 <= n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Scan x1
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t1);
        __m256 x1_low_to_high = _mm256_permute2f128_ps(x1, x1, 0x08);
        x1_low_to_high = _mm256_permute_ps(x1_low_to_high, 0xFF);
        x1 = _mm256_add_ps(x1, x1_low_to_high);
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Update carry for x2 from the end of x1
        __m256 v_carry_x2 = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry_x2 = _mm256_permute_ps(v_carry_x2, 0xFF);

        // Scan x2
        __m256 t2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x2 = _mm256_add_ps(x2, t2);
        __m256 t3 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x2 = _mm256_add_ps(x2, t3);
        __m256 x2_low_to_high = _mm256_permute2f128_ps(x2, x2, 0x08);
        x2_low_to_high = _mm256_permute_ps(x2_low_to_high, 0xFF);
        x2 = _mm256_add_ps(x2, x2_low_to_high);
        x2 = _mm256_add_ps(x2, v_carry_x2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Update global carry for next 16-element block
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