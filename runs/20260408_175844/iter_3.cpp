#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    if (n >= 8) {
        __m256 offset = _mm256_setzero_ps();
        
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(input + i);

            // Step 1: Intra-lane prefix sum (Kogge-Stone)
            // [a, b, c, d, e, f, g, h]
            __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, shift1);
            // [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
            __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, shift2);
            // [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]

            // Step 2: Correct the high lane (elements 4-7) using the sum of the low lane (element 3)
            __m256 low_sum = _mm256_permute2f128_ps(x, x, 0x00); // Broadcast low 128 to high
            low_sum = _mm256_permute_ps(low_sum, _MM_SHUFFLE(3, 3, 3, 3));
            
            // Mask out the low lane so we only add to the high lane
            __m256 high_mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            x = _mm256_add_ps(x, _mm256_and_ps(low_sum, high_mask));

            // Step 3: Add the offset from all previous blocks
            x = _mm256_add_ps(x, offset);

            // Store result
            _mm256_storeu_ps(output + i, x);

            // Step 4: Update offset for next block (broadcast the last element of current block)
            // The last element is at index 7.
            __m256 last_val_high = _mm256_permute2f128_ps(x, x, 0x11);
            offset = _mm256_permute_ps(last_val_high, _MM_SHUFFLE(3, 3, 3, 3));
        }
        
        // Update running_sum for the scalar tail
        if (i > 0) {
            running_sum = output[i - 1];
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}