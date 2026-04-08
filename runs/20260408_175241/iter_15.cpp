#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    __m256 v_running_sum = _mm256_setzero_ps();
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: Internal scan of the 8-float vector
        // [a, b, c, d, e, f, g, h] -> [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
        __m256 shift1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(2, 1, 0, 3));
        shift1 = _mm256_blend_ps(_mm256_setzero_ps(), shift1, 0xEE); // 0b11101110
        x = _mm256_add_ps(x, shift1);

        // [a, a+b, b+c, c+d, ...] -> [a, a+b, a+b+c, a+b+c+d, ...]
        __m256 shift2 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 3, 2));
        shift2 = _mm256_blend_ps(_mm256_setzero_ps(), shift2, 0xCC); // 0b11001100
        x = _mm256_add_ps(x, shift2);

        // Cross-lane propagation (low 128 to high 128)
        __m256 high_lane = _mm256_permute2f128_ps(x, x, 0x00); 
        __m256 shift4 = _mm256_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3));
        shift4 = _mm256_blend_ps(_mm256_setzero_ps(), shift4, 0xF0); // 0b11110000
        x = _mm256_add_ps(x, shift4);

        // Step 2: Add the running sum from previous blocks
        x = _mm256_add_ps(x, v_running_sum);
        _mm256_storeu_ps(&output[i], x);

        // Step 3: Update running sum for next iteration
        // Broadcast the last element of the current result
        __m256 last_lane = _mm256_permute2f128_ps(x, x, 0x11);
        v_running_sum = _mm256_shuffle_ps(last_lane, last_lane, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Tail handling
    float scalar_running_sum = (i == 0) ? 0.0f : output[i - 1];
    for (; i < n; ++i) {
        scalar_running_sum += input[i];
        output[i] = scalar_running_sum;
    }
}