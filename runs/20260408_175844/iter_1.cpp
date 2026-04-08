#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    // Scalar fallback for very small arrays
    if (n < 16) {
        float sum = input[0];
        output[0] = sum;
        for (size_t i = 1; i < n; ++i) {
            sum += input[i];
            output[i] = sum;
        }
        return;
    }

    float offset = 0.0f;
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Internal prefix sum within the register (Kogge-Stone)
        // [a, b, c, d, e, f, g, h]
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        // [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        // [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]
        
        // Cross-lane: add the sum of the low 128-bit lane to the high 128-bit lane
        __m256 low_sum = _mm256_permute2f128_ps(x, x, 0x00); // Broadcast low 128 to both
        __m256 high_part = _mm256_permute_ps(low_sum, 0xFF); // Get 4th element (index 3)
        
        // Mask to only add to the upper lane
        __m256 mask = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        x = _mm256_add_ps(x, _mm256_and_ps(mask, high_part));

        // Add the carry from the previous block
        __m256 v_offset = _mm256_set1_ps(offset);
        x = _mm256_add_ps(x, v_offset);

        _mm256_storeu_ps(&output[i], x);

        // Extract the last element to be the offset for the next block
        // Using a shuffle/extract is faster than memory access
        __m128 high_lane = _mm256_extractf128_ps(x, 1);
        offset = _mm_cvtss_f32(_mm_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3)));
    }

    // Final scalar cleanup
    for (; i < n; ++i) {
        offset += input[i];
        output[i] = offset;
    }
}