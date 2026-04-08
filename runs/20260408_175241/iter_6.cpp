#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 16 (2 x 8) to hide instruction latency
    for (; i + 15 < n; i += 16) {
        __m256 in0 = _mm256_loadu_ps(&input[i]);
        __m256 in1 = _mm256_loadu_ps(&input[i + 8]);

        // Scan first 8 elements
        __m256 shift4_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in0), 4));
        in0 = _mm256_add_ps(in0, shift4_0);
        __m256 shift8_0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in0), 8));
        in0 = _mm256_add_ps(in0, shift8_0);
        __m256 low_sum0 = _mm256_permute2f128_ps(in0, in0, 0x00);
        low_sum0 = _mm256_permute_ps(low_sum0, 0xFF);
        in0 = _mm256_add_ps(in0, _mm256_blend_ps(_mm256_setzero_ps(), low_sum0, 0xF0));
        
        // Apply running sum to first block
        in0 = _mm256_add_ps(in0, _mm256_set1_ps(running_sum));
        _mm256_storeu_ps(&output[i], in0);

        // Extract last element of first block to use as carry for second block
        __m128 high_lane0 = _mm256_extractf128_ps(in0, 1);
        float carry_mid = _mm_cvtss_f32(_mm_permute_ps(high_lane0, 0xFF));

        // Scan second 8 elements
        __m256 shift4_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in1), 4));
        in1 = _mm256_add_ps(in1, shift4_1);
        __m256 shift8_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in1), 8));
        in1 = _mm256_add_ps(in1, shift8_1);
        __m256 low_sum1 = _mm256_permute2f128_ps(in1, in1, 0x00);
        low_sum1 = _mm256_permute_ps(low_sum1, 0xFF);
        in1 = _mm256_add_ps(in1, _mm256_blend_ps(_mm256_setzero_ps(), low_sum1, 0xF0));

        // Apply carry_mid to second block
        in1 = _mm256_add_ps(in1, _mm256_set1_ps(carry_mid));
        _mm256_storeu_ps(&output[i + 8], in1);

        // Update running_sum for next iteration
        __m128 high_lane1 = _mm256_extractf128_ps(in1, 1);
        running_sum = _mm_cvtss_f32(_mm_permute_ps(high_lane1, 0xFF));
    }

    // Process remaining blocks of 8
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);
        __m256 shift4 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, shift4);
        __m256 shift8 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, shift8);
        __m256 low_sum = _mm256_permute2f128_ps(data, data, 0x00);
        low_sum = _mm256_permute_ps(low_sum, 0xFF);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), low_sum, 0xF0));
        
        data = _mm256_add_ps(data, _mm256_set1_ps(running_sum));
        _mm256_storeu_ps(&output[i], data);

        __m128 high_lane = _mm256_extractf128_ps(data, 1);
        running_sum = _mm_cvtss_f32(_mm_permute_ps(high_lane, 0xFF));
    }

    // Tail handling
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}