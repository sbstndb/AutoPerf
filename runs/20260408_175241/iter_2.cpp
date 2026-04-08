#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 8 <= n; i += 8) {
        // Load 8 elements: [a, b, c, d, e, f, g, h]
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: [a, a+b, b+c, c+d, d+e, e+f, f+g, g+h]
        __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, shift1);

        // Step 2: [a, a+b, a+b+c, a+b+c+d, b+c+d+e, c+d+e+f, d+e+f+g, e+f+g+h]
        __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, shift2);

        // Step 3: Cross the 128-bit lane boundary
        // Extract high lane, add sum of low lane to all elements in high lane
        __m256 low_to_high = _mm256_permute2f128_ps(x, x, 0x00); // [L, L]
        // Broadcast the 4th element (index 3) of the low lane to all
        __m256 sum_low = _mm256_shuffle_ps(low_to_high, low_to_high, _MM_SHUFFLE(3, 3, 3, 3));
        
        // Mask to only add sum_low to the high lane (elements 4-7)
        // We can use a blend or just zero out the low part of the addition
        __m256 high_lane_mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        x = _mm256_add_ps(x, _mm256_and_ps(sum_low, high_lane_mask));

        // Add the running sum from previous blocks to all 8 elements
        __m256 v_running = _mm256_set1_ps(running_sum);
        x = _mm256_add_ps(x, v_running);

        // Store results
        _mm256_storeu_ps(&output[i], x);

        // Update running sum from the last element of the current block
        // We extract the 7th index (last element)
        float last_val;
        _mm_store_ss(&last_val, _mm256_extractf128_ps(x, 1)); // This gets high lane
        // Actually, simpler to just grab it from the output array to avoid complex extraction
        running_sum = output[i + 7];
    }

    // Scalar tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}