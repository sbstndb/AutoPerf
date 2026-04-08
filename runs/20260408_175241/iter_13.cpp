#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    __m256 v_running_sum = _mm256_setzero_ps();
    size_t i = 0;

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 in = _mm256_loadu_ps(&input[i]);

        // Step 1: Intra-lane prefix sum (0-3 and 4-7)
        // [a, b, c, d | e, f, g, h]
        __m256 shift1 = _mm256_permute_ps(in, _MM_SHUFFLE(2, 1, 0, 0));
        shift1 = _mm256_blend_ps(shift1, _mm256_setzero_ps(), 0x11); 
        __m256 sum = _mm256_add_ps(in, shift1);

        __m256 shift2 = _mm256_permute_ps(sum, _MM_SHUFFLE(1, 0, 0, 0));
        shift2 = _mm256_blend_ps(shift2, _mm256_setzero_ps(), 0x33);
        sum = _mm256_add_ps(sum, shift2);

        // Step 2: Inter-lane carry (propagate sum of low lane to high lane)
        __m256 low_to_high = _mm256_permute2f128_ps(sum, sum, 0x00); 
        low_to_high = _mm256_permute_ps(low_to_high, 0xFF);
        sum = _mm256_add_ps(sum, _mm256_blend_ps(_mm256_setzero_ps(), low_to_high, 0xF0));

        // Step 3: Add the running sum from previous blocks
        sum = _mm256_add_ps(sum, v_running_sum);
        _mm256_storeu_ps(&output[i], sum);

        // Step 4: Prepare running sum for next iteration (broadcast last element)
        __m256 last_val = _mm256_permute2f128_ps(sum, sum, 0x11);
        v_running_sum = _mm256_permute_ps(last_val, 0xFF);
    }

    // Tail handling
    float scalar_running_sum = _mm_cvtss_f32(_mm256_castps256_ps128(v_running_sum));
    if (i > 0) {
        // The loop above leaves the last sum in all lanes of v_running_sum
        // but we need to extract it correctly.
        scalar_running_sum = output[i-1];
    }

    for (; i < n; ++i) {
        scalar_running_sum += input[i];
        output[i] = scalar_running_sum;
    }
}