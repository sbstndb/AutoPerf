#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x 8-float AVX blocks)
    for (; i + 16 <= n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane scan for both blocks (Parallel)
        // Shift 4 bytes (1 float)
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        x2 = _mm256_add_ps(x2, t0_2);

        // Shift 8 bytes (2 floats)
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x1 = _mm256_add_ps(x1, t1_1);
        x2 = _mm256_add_ps(x2, t1_2);

        // Step 2: Cross-lane scan (Lower 128 to Upper 128)
        __m256 lth1 = _mm256_permute2f128_ps(x1, x1, 0x08);
        __m256 lth2 = _mm256_permute2f128_ps(x2, x2, 0x08);
        lth1 = _mm256_shuffle_ps(lth1, lth1, _MM_SHUFFLE(3, 3, 3, 3));
        lth2 = _mm256_shuffle_ps(lth2, lth2, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, lth1);
        x2 = _mm256_add_ps(x2, lth2);

        // Step 3: Apply carry from previous 16-block to x1
        x1 = _mm256_add_ps(x1, v_carry);

        // Step 4: Calculate carry for x2 from the end of x1
        __m128 x1_high = _mm256_extractf128_ps(x1, 1);
        __m128 v_carry2_128 = _mm_shuffle_ps(x1_high, x1_high, _MM_SHUFFLE(3, 3, 3, 3));
        __m256 v_carry2 = _mm256_set_m128(v_carry2_128, v_carry2_128);
        
        x2 = _mm256_add_ps(x2, v_carry2);

        // Store results
        _mm256_storeu_ps(&output[i], x1);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Step 5: Update carry for next iteration from end of x2
        __m128 x2_high = _mm256_extractf128_ps(x2, 1);
        __m128 v_carry_next_128 = _mm_shuffle_ps(x2_high, x2_high, _MM_SHUFFLE(3, 3, 3, 3));
        v_carry = _mm256_set_m128(v_carry_next_128, v_carry_next_128);
    }

    // Process remaining 8-element block
    if (i + 8 <= n) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 lth = _mm256_permute2f128_ps(x, x, 0x08);
        lth = _mm256_shuffle_ps(lth, lth, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, lth);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        i += 8;
    }

    // Final scalar tail
    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}