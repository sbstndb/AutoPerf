#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    __m256 running_sum_v = _mm256_setzero_ps();
    size_t i = 0;

    // Process in blocks of 16 (2 x 8)
    for (; i + 15 < n; i += 16) {
        // BLOCK 1
        __m256 in0 = _mm256_loadu_ps(&input[i]);
        
        // Intra-register scan for 8 elements
        __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in0), 4));
        in0 = _mm256_add_ps(in0, shift1);
        __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in0), 8));
        in0 = _mm256_add_ps(in0, shift2);
        __m256 low_to_high = _mm256_permute2f128_ps(in0, in0, 0x00); 
        // Broadcast the 4th element of the low lane to all elements of the high lane
        __m256 carry8 = _mm256_permute_ps(low_to_high, 0xFF);
        in0 = _mm256_add_ps(in0, _mm256_blend_ps(_mm256_setzero_ps(), carry8, 0xF0));
        
        // Add previous running sum
        in0 = _mm256_add_ps(in0, running_sum_v);
        _mm256_storeu_ps(&output[i], in0);

        // Prepare running_sum for block 2 (broadcast last element of in0)
        __m256 last_val0 = _mm256_permute2f128_ps(in0, in0, 0x11);
        running_sum_v = _mm256_permute_ps(last_val0, 0xFF);

        // BLOCK 2
        __m256 in1 = _mm256_loadu_ps(&input[i + 8]);
        __m256 shift1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in1), 4));
        in1 = _mm256_add_ps(in1, shift1_1);
        __m256 shift2_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(in1), 8));
        in1 = _mm256_add_ps(in1, shift2_1);
        __m256 low_to_high1 = _mm256_permute2f128_ps(in1, in1, 0x00);
        __m256 carry8_1 = _mm256_permute_ps(low_to_high1, 0xFF);
        in1 = _mm256_add_ps(in1, _mm256_blend_ps(_mm256_setzero_ps(), carry8_1, 0xF0));

        in1 = _mm256_add_ps(in1, running_sum_v);
        _mm256_storeu_ps(&output[i + 8], in1);

        // Prepare running_sum for next iteration
        __m256 last_val1 = _mm256_permute2f128_ps(in1, in1, 0x11);
        running_sum_v = _mm256_permute_ps(last_val1, 0xFF);
    }

    // Process remaining blocks of 8
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);
        __m256 s1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, s1);
        __m256 s2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, s2);
        __m256 lth = _mm256_permute2f128_ps(data, data, 0x00);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(lth, 0xFF), 0xF0));
        
        data = _mm256_add_ps(data, running_sum_v);
        _mm256_storeu_ps(&output[i], data);

        __m256 lv = _mm256_permute2f128_ps(data, data, 0x11);
        running_sum_v = _mm256_permute_ps(lv, 0xFF);
    }

    // Scalar tail
    float carry = _mm256_cvtss_f32(running_sum_v);
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}