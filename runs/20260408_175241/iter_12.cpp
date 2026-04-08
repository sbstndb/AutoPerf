#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);

        // Step 1: Intra-lane prefix sum (0-3 and 4-7 independently)
        // [a, b, c, d | e, f, g, h]
        __m256 shift1 = _mm256_permute_ps(data, _MM_SHUFFLE(2, 1, 0, 0));
        shift1 = _mm256_blend_ps(shift1, _mm256_setzero_ps(), 0x11); // [0, a, b, c | 0, e, f, g]
        data = _mm256_add_ps(data, shift1);

        __m256 shift2 = _mm256_permute_ps(data, _MM_SHUFFLE(1, 0, 0, 0));
        shift2 = _mm256_blend_ps(shift2, _mm256_setzero_ps(), 0x33); // [0, 0, a, a+b | 0, 0, e, e+f]
        data = _mm256_add_ps(data, shift2);

        // Step 2: Cross-lane fixup (add sum of low lane to high lane)
        // Extract sum of first 4 elements (index 3)
        __m256 low_sum = _mm256_permute2f128_ps(data, data, 0x00); 
        low_sum = _mm256_permute_ps(low_sum, 0xFF); 
        // Add to high lane only
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), low_sum, 0xF0));

        // Step 3: Apply the running sum from previous blocks
        __m256 carry = _mm256_set1_ps(running_sum);
        data = _mm256_add_ps(data, carry);

        _mm256_storeu_ps(&output[i], data);

        // Step 4: Update running_sum for next iteration
        // Extract the last element of the current result
        __m128 high_lane = _mm256_extractf128_ps(data, 1);
        running_sum = _mm_cvtss_f32(_mm_permute_ps(high_lane, 0xFF));
    }

    // Scalar tail for remaining elements
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}