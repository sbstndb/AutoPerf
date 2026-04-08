#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    if (n < 16) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            acc += input[i];
            output[i] = acc;
        }
        return;
    }

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 16 (2x8) to improve ILP
    for (; i + 15 < n; i += 16) {
        for (int k = 0; k < 2; ++k) {
            __m256 data = _mm256_loadu_ps(&input[i + k * 8]);

            // Kogge-Stone style scan within 8-wide SIMD
            // [a, b, c, d, e, f, g, h]
            __m256 shifted1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
            data = _mm256_add_ps(data, shifted1); 
            
            __m256 shifted2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
            data = _mm256_add_ps(data, shifted2);

            // Cross-lane fixup: add sum of low 128-bit lane to high 128-bit lane
            __m256 low_sum = _mm256_permute2f128_ps(data, data, 0x00); 
            low_sum = _mm256_permute_ps(low_sum, 0xFF); // Broadcast element 3
            data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), low_sum, 0xF0));

            // Add carry from previous iterations
            __m256 carry = _mm256_set1_ps(running_sum);
            data = _mm256_add_ps(data, carry);

            _mm256_storeu_ps(&output[i + k * 8], data);

            // Extract the last element (index 7) to update running_sum
            __m128 high_lane = _mm256_extractf128_ps(data, 1);
            running_sum = _mm_cvtss_f32(_mm_permute_ps(high_lane, 0xFF));
        }
    }

    // Handle remaining blocks of 8
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);
        __m256 shifted1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, shifted1); 
        __m256 shifted2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, shifted2);
        __m256 low_sum = _mm256_permute2f128_ps(data, data, 0x00); 
        low_sum = _mm256_permute_ps(low_sum, 0xFF);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), low_sum, 0xF0));
        data = _mm256_add_ps(data, _mm256_set1_ps(running_sum));
        _mm256_storeu_ps(&output[i], data);
        __m128 high_lane = _mm256_extractf128_ps(data, 1);
        running_sum = _mm_cvtss_f32(_mm_permute_ps(high_lane, 0xFF));
    }

    // Final tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}