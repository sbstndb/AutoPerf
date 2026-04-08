#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    if (n < 8) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            acc += input[i];
            output[i] = acc;
        }
        return;
    }

    __m256 curr_carry = _mm256_setzero_ps();
    size_t i = 0;

    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        // Block 1
        __m256 data1 = _mm256_loadu_ps(&input[i]);
        __m256 shift1_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data1), 4));
        data1 = _mm256_add_ps(data1, shift1_1);
        __m256 shift1_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data1), 8));
        data1 = _mm256_add_ps(data1, shift1_2);
        __m256 low_sum1 = _mm256_permute2f128_ps(data1, data1, 0x00);
        low_sum1 = _mm256_permute_ps(low_sum1, 0xFF);
        data1 = _mm256_add_ps(data1, _mm256_blend_ps(_mm256_setzero_ps(), low_sum1, 0xF0));
        data1 = _mm256_add_ps(data1, curr_carry);
        _mm256_storeu_ps(&output[i], data1);
        
        // Broadcast last element of data1 to curr_carry
        curr_carry = _mm256_permute2f128_ps(data1, data1, 0x11);
        curr_carry = _mm256_permute_ps(curr_carry, 0xFF);

        // Block 2
        __m256 data2 = _mm256_loadu_ps(&input[i + 8]);
        __m256 shift2_1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data2), 4));
        data2 = _mm256_add_ps(data2, shift2_1);
        __m256 shift2_2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data2), 8));
        data2 = _mm256_add_ps(data2, shift2_2);
        __m256 low_sum2 = _mm256_permute2f128_ps(data2, data2, 0x00);
        low_sum2 = _mm256_permute_ps(low_sum2, 0xFF);
        data2 = _mm256_add_ps(data2, _mm256_blend_ps(_mm256_setzero_ps(), low_sum2, 0xF0));
        data2 = _mm256_add_ps(data2, curr_carry);
        _mm256_storeu_ps(&output[i + 8], data2);
        
        // Broadcast last element of data2 to curr_carry
        curr_carry = _mm256_permute2f128_ps(data2, data2, 0x11);
        curr_carry = _mm256_permute_ps(curr_carry, 0xFF);
    }

    // Handle remaining 8-float blocks
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);
        __m256 s1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, s1);
        __m256 s2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, s2);
        __m256 ls = _mm256_permute2f128_ps(data, data, 0x00);
        ls = _mm256_permute_ps(ls, 0xFF);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), ls, 0xF0));
        data = _mm256_add_ps(data, curr_carry);
        _mm256_storeu_ps(&output[i], data);
        
        curr_carry = _mm256_permute2f128_ps(data, data, 0x11);
        curr_carry = _mm256_permute_ps(curr_carry, 0xFF);
    }

    // Final tail
    float scalar_carry = 0.0f;
    if (i > 0) {
        scalar_carry = output[i - 1];
    } else {
        scalar_carry = _mm_cvtss_f32(_mm256_castps256_ps128(curr_carry));
    }

    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}