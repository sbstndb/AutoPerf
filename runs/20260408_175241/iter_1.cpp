#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    // Handle small arrays or the first element
    float acc = 0.0f;
    size_t i = 0;

    // SIMD processing: 8 floats at a time (AVX2)
    if (n >= 8) {
        __m256 v_acc = _mm256_setzero_ps();
        
        // Mask for shifting: [0, 0, 1, 2, 3, 4, 5, 6]
        // We use shuffles to perform intra-register prefix sums
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: Shift and add (Kogge-Stone)
            // [a, b, c, d, e, f, g, h] -> [a, a+b, b+c, c+d, d+e, e+f, f+g, g+h]
            __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, shift1);

            // Step 2: Shift and add by 2
            __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, shift2);

            // Step 3: Cross 128-bit lane shuffle and add
            // The bslli only works within 128-bit lanes. We need to bridge the gap.
            __m256 low_to_high = _mm256_permute2f128_ps(x, x, 0x08); // [L, L]
            // We only want the highest value of the low lane added to all high lane values
            __m256 high_sum = _mm256_shuffle_ps(low_to_high, low_to_high, _MM_SHUFFLE(3, 3, 3, 3));
            // Mask out the low lane so we only add to the high lane
            __m256 mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            x = _mm256_add_ps(x, _mm256_and_ps(mask, high_sum));

            // Step 4: Add the carry from the previous block
            x = _mm256_add_ps(x, v_acc);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Update carry: broadcast the last element of the current computed block
            // The last element is at index 7.
            __m256 v_last = _mm256_permute2f128_ps(x, x, 0x11);
            v_acc = _mm256_shuffle_ps(v_last, v_last, _MM_SHUFFLE(3, 3, 3, 3));
        }
        
        // Extract the last carry to continue with scalar tail
        if (i > 0) {
            acc = output[i - 1];
        }
    }

    // Scalar tail for remaining elements
    for (; i < n; ++i) {
        acc += input[i];
        output[i] = acc;
    }
}