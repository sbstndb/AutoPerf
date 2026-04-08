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
        __m256 t0_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t0_1);
        __m256 t1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t1_1);

        // Block 2: Intra-lane scan (Independent of Block 1)
        __m256 t0_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x2 = _mm256_add_ps(x2, t0_2);
        __m256 t1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x2 = _mm256_add_ps(x2, t1_2);

        // Block 1: Cross-lane and Carry
        __m256 lth1 = _mm256_permute2f128_ps(x1, x1, 0x08);
        lth1 = _mm256_permute_ps(lth1, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, lth1);
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Update carry for Block 2 using Block 1's last element
        __m256 v_carry2 = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry2 = _mm256_permute_ps(v_carry2, _MM_SHUFFLE(3, 3, 3, 3));

        // Block 2: Cross-lane and Carry
        __m256 lth2 = _mm256_permute2f128_ps(x2, x2, 0x08);
        lth2 = _mm256_permute_ps(lth2, _MM_SHUFFLE(3, 3, 3, 3));
        x2 = _mm256_add_ps(x2, lth2);
        x2 = _mm256_add_ps(x2, v_carry2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Update carry for next iteration
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 lth = _mm256_permute2f128_ps(x, x, 0x08);
        lth = _mm256_permute_ps(lth, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, lth);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    float carry = (i > 0) ? output[i - 1] : 0.0f;
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}