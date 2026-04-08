#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(&input[i]);
        __m256 x1 = _mm256_loadu_ps(&input[i + 8]);

        // Scan x0
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x0), 4));
        x0 = _mm256_add_ps(x0, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x0), 8));
        x0 = _mm256_add_ps(x0, t1);
        
        // Propagate low 128-bit lane to high 128-bit lane for x0
        __m256 x0_low = _mm256_permute2f128_ps(x0, x0, 0x00);
        __m256 x0_sum = _mm256_shuffle_ps(x0_low, x0_low, _MM_SHUFFLE(3, 3, 3, 3));
        x0 = _mm256_add_ps(x0, _mm256_set_m128(_mm256_castps256_ps128(x0_sum), _mm_setzero_ps()));
        
        // Apply global carry to x0
        x0 = _mm256_add_ps(x0, v_carry);

        // Scan x1
        __m256 t2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t2);
        __m256 t3 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t3);

        // Propagate low 128-bit lane to high 128-bit lane for x1
        __m256 x1_low = _mm256_permute2f128_ps(x1, x1, 0x00);
        __m256 x1_sum = _mm256_shuffle_ps(x1_low, x1_low, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, _mm256_set_m128(_mm256_castps256_ps128(x1_sum), _mm_setzero_ps()));

        // Calculate carry for x1 from the end of x0
        __m256 v_carry_x1 = _mm256_permute2f128_ps(x0, x0, 0x11);
        v_carry_x1 = _mm256_shuffle_ps(v_carry_x1, v_carry_x1, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, v_carry_x1);

        _mm256_storeu_ps(&output[i], x0);
        _mm256_storeu_ps(&output[i + 8], x1);

        // Update global carry for next iteration
        __m256 v_carry_next = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry = _mm256_shuffle_ps(v_carry_next, v_carry_next, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 x_low = _mm256_permute2f128_ps(x, x, 0x00);
        __m256 x_sum = _mm256_shuffle_ps(x_low, x_low, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, _mm256_set_m128(_mm256_castps256_ps128(x_sum), _mm_setzero_ps()));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        __m256 v_carry_next = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_shuffle_ps(v_carry_next, v_carry_next, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}