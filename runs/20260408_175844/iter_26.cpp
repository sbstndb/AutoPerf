#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2 x YMM)
    for (; i + 15 < n; i += 16) {
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Step 1: Intra-lane scan for x1
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t1);
        
        // Step 2: Cross-lane for x1
        __m256 x1_low = _mm256_permute2f128_ps(x1, x1, 0x20); 
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(x1_low, 0xFF), 0xF0));
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Step 3: Extract carry for x2 from x1
        __m256 v_carry_x2 = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry_x2 = _mm256_permute_ps(v_carry_x2, 0xFF);

        // Step 4: Intra-lane scan for x2
        __m256 t2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x2 = _mm256_add_ps(x2, t2);
        __m256 t3 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x2 = _mm256_add_ps(x2, t3);

        // Step 5: Cross-lane for x2
        __m256 x2_low = _mm256_permute2f128_ps(x2, x2, 0x20);
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(x2_low, 0xFF), 0xF0));
        x2 = _mm256_add_ps(x2, v_carry_x2);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Step 6: Update global carry
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
        __m256 x_low = _mm256_permute2f128_ps(x, x, 0x20);
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(x_low, 0xFF), 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Scalar tail
    float carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}