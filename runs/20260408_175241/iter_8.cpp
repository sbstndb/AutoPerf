#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    if (n >= 8) {
        __m256 v_running_sum = _mm256_setzero_ps();
        
        for (; i + 7 < n; i += 8) {
            // Load 8 floats
            __m256 data = _mm256_loadu_ps(&input[i]);

            // Step 1: Intra-lane prefix sum (Kogge-Stone)
            // Shift and add by 1 (4 bytes)
            __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
            data = _mm256_add_ps(data, shift1);

            // Shift and add by 2 (8 bytes)
            __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
            data = _mm256_add_ps(data, shift2);

            // Step 2: Cross-lane propagation
            // Broadcast the last element of the lower 128-bit lane to the upper 128-bit lane
            __m256 lane_sum = _mm256_permute2f128_ps(data, data, 0x00); // Copy low lane to high
            lane_sum = _mm256_permute_ps(lane_sum, 0xFF);               // Broadcast index 3 to all
            
            // Mask: Add lane_sum only to the upper 128 bits (elements 4-7)
            data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), lane_sum, 0xF0));

            // Step 3: Apply the global running sum from previous iterations
            data = _mm256_add_ps(data, v_running_sum);
            
            // Store results
            _mm256_storeu_ps(&output[i], data);

            // Step 4: Update running sum for next iteration
            // Extract the last element (index 7) of the current result
            __m128 last_lane = _mm256_extractf128_ps(data, 1);
            v_running_sum = _mm256_broadcast_ss((float*)&last_lane + 3);
        }
        
        // Sync scalar running sum for the tail
        __m128 final_lane = _mm256_extractf128_ps(v_running_sum, 0);
        running_sum = _mm_cvtss_f32(final_lane);
    }

    // Tail handling (Scalar)
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}