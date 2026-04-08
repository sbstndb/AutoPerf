#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    if (n >= 8) {
        __m256 v_running_sum = _mm256_setzero_ps();
        
        // Unroll by 2 to process 16 elements per iteration
        for (; i + 15 < n; i += 16) {
            // Block 1
            __m256 data1 = _mm256_loadu_ps(&input[i]);
            // Intra-lane scan (Kogge-Stone style)
            __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data1), 4));
            data1 = _mm256_add_ps(data1, shift1);
            shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data1), 8));
            data1 = _mm256_add_ps(data1, shift1);
            // Cross-lane propagation
            __m256 high1 = _mm256_permute2f128_ps(data1, data1, 0x00);
            high1 = _mm256_permute_ps(high1, 0xFF);
            data1 = _mm256_add_ps(data1, _mm256_blend_ps(_mm256_setzero_ps(), high1, 0xF0));
            // Add carry from previous block
            data1 = _mm256_add_ps(data1, _mm256_set1_ps(running_sum));
            _mm256_storeu_ps(&output[i], data1);
            
            // Extract carry for Block 2
            __m128 last1 = _mm256_extractf128_ps(data1, 1);
            running_sum = _mm_cvtss_f32(_mm_permute_ps(last1, 0xFF));

            // Block 2
            __m256 data2 = _mm256_loadu_ps(&input[i + 8]);
            __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data2), 4));
            data2 = _mm256_add_ps(data2, shift2);
            shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data2), 8));
            data2 = _mm256_add_ps(data2, shift2);
            __m256 high2 = _mm256_permute2f128_ps(data2, data2, 0x00);
            high2 = _mm256_permute_ps(high2, 0xFF);
            data2 = _mm256_add_ps(data2, _mm256_blend_ps(_mm256_setzero_ps(), high2, 0xF0));
            data2 = _mm256_add_ps(data2, _mm256_set1_ps(running_sum));
            _mm256_storeu_ps(&output[i + 8], data2);

            // Extract carry for next iteration
            __m128 last2 = _mm256_extractf128_ps(data2, 1);
            running_sum = _mm_cvtss_f32(_mm_permute_ps(last2, 0xFF));
        }

        // Handle remaining 8-element blocks
        for (; i + 7 < n; i += 8) {
            __m256 data = _mm256_loadu_ps(&input[i]);
            __m256 shift = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
            data = _mm256_add_ps(data, shift);
            shift = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
            data = _mm256_add_ps(data, shift);
            __m256 high = _mm256_permute2f128_ps(data, data, 0x00);
            high = _mm256_permute_ps(high, 0xFF);
            data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), high, 0xF0));
            data = _mm256_add_ps(data, _mm256_set1_ps(running_sum));
            _mm256_storeu_ps(&output[i], data);
            
            __m128 last = _mm256_extractf128_ps(data, 1);
            running_sum = _mm_cvtss_f32(_mm_permute_ps(last, 0xFF));
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}