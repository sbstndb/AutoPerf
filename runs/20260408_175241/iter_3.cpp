#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    if (n < 16) {
        output[0] = input[0];
        for (size_t i = 1; i < n; ++i) {
            output[i] = output[i - 1] + input[i];
        }
        return;
    }

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);

        // Step 1: Internal scan of the 8-float vector
        // [a, b, c, d, e, f, g, h]
        __m256 shifted = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, shifted); 
        // [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
        
        shifted = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, shifted);
        // [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]

        // Fix the high lane (elements 4-7) by adding the sum of the low lane (element 3)
        __m256 low_sum = _mm256_permute2f128_ps(data, data, 0x00); // [L, L]
        low_sum = _mm256_permute_ps(low_sum, 0xFF); // Broadcast element 3 to all
        
        // Mask to only add to the high lane
        __m256 mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), low_sum, 0xF0));

        // Step 2: Add the running sum from previous iterations
        __m256 carry = _mm256_set1_ps(running_sum);
        data = _mm256_add_ps(data, carry);

        _mm256_storeu_ps(&output[i], data);

        // Update running_sum for the next block (extract last element)
        // We use a trick to get the 8th element without extra instructions
        alignas(32) float temp[8];
        _mm256_store_ps(temp, data);
        running_sum = temp[7];
    }

    // Tail handling
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}