#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    if (n >= 8) {
        __m256 offset = _mm256_setzero_ps();
        size_t i = 0;
        size_t n_vec = n & ~7; // Process in blocks of 8

        for (; i < n_vec; i += 8) {
            // Load 8 floats: [a, b, c, d, e, f, g, h]
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: Internal prefix sum within the 8-lane register
            // Shift and add: [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
            __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, t0);
            
            // Shift and add: [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]
            __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, t1);

            // Step 2: Propagate the sum of the lower 128-bit lane to the upper lane
            // Extract sum of first 4 elements and broadcast to upper 4
            __m256 low_sum = _mm256_permute2f128_ps(x, x, 0x00); // [L, L] where L is lower 4
            // We only want to add the 4th element of the lower lane to all elements of the upper lane
            __m256 high_offset = _mm256_shuffle_ps(low_sum, low_sum, _MM_SHUFFLE(3, 3, 3, 3));
            // Mask out the lower lane so we only add to the upper lane
            __m256 mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            x = _mm256_add_ps(x, _mm256_and_ps(mask, high_offset));

            // Step 3: Add the carry-over offset from all previous blocks
            x = _mm256_add_ps(x, offset);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Step 4: Update offset for next block (broadcast the last element of current result)
            offset = _mm256_set1_ps(output[i + 7]);
        }

        // Handle remaining elements
        for (; i < n; ++i) {
            output[i] = output[i - 1] + input[i];
        }
    } else {
        // Small arrays: pure scalar
        output[0] = input[0];
        for (size_t i = 1; i < n; ++i) {
            output[i] = output[i - 1] + input[i];
        }
    }
}