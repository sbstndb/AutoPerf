#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    if (n >= 16) {
        const __m256 zero = _mm256_setzero_ps();
        for (; i + 15 < n; i += 16) {
            // Load two blocks of 8 floats
            __m256 d0 = _mm256_loadu_ps(&input[i]);
            __m256 d1 = _mm256_loadu_ps(&input[i + 8]);

            // Intra-lane scan for block 0
            __m256 s1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d0), 4));
            d0 = _mm256_add_ps(d0, s1);
            __m256 s2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d0), 8));
            d0 = _mm256_add_ps(d0, s2);
            // Cross-lane fixup for block 0
            __m256 c0 = _mm256_permute2f128_ps(d0, d0, 0x00); 
            c0 = _mm256_permute_ps(c0, 0xFF);
            d0 = _mm256_add_ps(d0, _mm256_blend_ps(zero, c0, 0xF0));

            // Intra-lane scan for block 1
            __m256 s3 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d1), 4));
            d1 = _mm256_add_ps(d1, s3);
            __m256 s4 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d1), 8));
            d1 = _mm256_add_ps(d1, s4);
            // Cross-lane fixup for block 1
            __m256 c1 = _mm256_permute2f128_ps(d1, d1, 0x00);
            c1 = _mm256_permute_ps(c1, 0xFF);
            d1 = _mm256_add_ps(d1, _mm256_blend_ps(zero, c1, 0xF0));

            // Apply running sum to block 0
            __m256 v_run = _mm256_set1_ps(running_sum);
            d0 = _mm256_add_ps(d0, v_run);
            _mm256_storeu_ps(&output[i], d0);

            // Apply block 0's total sum to block 1
            __m256 v_last0 = _mm256_permute2f128_ps(d0, d0, 0x11);
            v_last0 = _mm256_permute_ps(v_last0, 0xFF);
            d1 = _mm256_add_ps(d1, v_last0);
            _mm256_storeu_ps(&output[i + 8], d1);

            // Update running_sum for next 16-float iteration
            __m128 high = _mm256_extractf128_ps(d1, 1);
            running_sum = _mm_cvtss_f32(_mm_permute_ps(high, 0xFF));
        }
    }

    // Handle remaining blocks of 8
    for (; i + 7 < n; i += 8) {
        __m256 d = _mm256_loadu_ps(&input[i]);
        __m256 s1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d), 4));
        d = _mm256_add_ps(d, s1);
        __m256 s2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(d), 8));
        d = _mm256_add_ps(d, s2);
        __m256 c = _mm256_permute2f128_ps(d, d, 0x00);
        c = _mm256_permute_ps(c, 0xFF);
        d = _mm256_add_ps(d, _mm256_blend_ps(_mm256_setzero_ps(), c, 0xF0));
        d = _mm256_add_ps(d, _mm256_set1_ps(running_sum));
        _mm256_storeu_ps(&output[i], d);
        __m128 high = _mm256_extractf128_ps(d, 1);
        running_sum = _mm_cvtss_f32(_mm_permute_ps(high, 0xFF));
    }

    // Final tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}