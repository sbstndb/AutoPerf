#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    float offset = 0.0f;

    // Process in blocks of 8 using AVX2
    if (n >= 8) {
        __m256 v_offset = _mm256_setzero_ps();
        
        for (; i <= n - 8; i += 8) {
            // Load 8 elements
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: [a, b, c, d, e, f, g, h] -> [a, a+b, b+c, c+d, d+e, e+f, f+g, g+h]
            __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, shift1);

            // Step 2: [a, a+b, a+b+c, b+c+d, ...]
            __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, shift2);

            // Step 3: Handle the cross-128-bit lane dependency
            // Extract the high element of the low 128-bit lane (index 3)
            __m256 high_of_low = _mm256_permute2f128_ps(x, x, 0x00); // Low lane to both
            high_of_low = _mm256_permute_ps(high_of_low, 0xFF);      // Broadcast index 3
            
            // Blend: only add the high_of_low to the upper 128-bit lane
            __m256 mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            __m256 shift4 = _mm256_and_ps(mask, high_of_low);
            x = _mm256_add_ps(x, shift4);

            // Add the carry-over from the previous 8-element block
            x = _mm256_add_ps(x, v_offset);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Prepare offset for next iteration: broadcast the last element (index 7)
            __m256 last_val = _mm256_permute2f128_ps(x, x, 0x11); // High lane to both
            v_offset = _mm256_permute_ps(last_val, 0xFF);         // Broadcast index 7
        }
        
        // Extract scalar offset for the remaining elements
        float last_block[8];
        _mm256_storeu_ps(last_block, v_offset);
        offset = last_block[0];
    }

    // Scalar tail for n < 8 or remaining elements
    for (; i < n; ++i) {
        offset += input[i];
        output[i] = offset;
    }
}