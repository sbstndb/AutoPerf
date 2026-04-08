#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    if (n >= 8) {
        // Pre-calculate masks to avoid reloading in the loop
        const __m256 mask_low4 = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));

        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: Shift by 1 (Internal to 128-bit lanes)
            // [a, b, c, d | e, f, g, h] -> [0, a, b, c | 0, e, f, g]
            __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 0));
            t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE); // 0b11101110
            x = _mm256_add_ps(x, t0);

            // Step 2: Shift by 2 (Internal to 128-bit lanes)
            // [a, ab, abc, abcd | ...] -> [0, 0, a, ab | ...]
            __m256 t1 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 0, 0));
            t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC); // 0b11001100
            x = _mm256_add_ps(x, t1);

            // Step 3: Shift by 4 (Cross 128-bit lanes)
            // High lane += broadcast(last element of low lane)
            __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20); // Low to High, Zero to Low
            // We only need the sum of the first 4 to be added to all of the second 4
            t2 = _mm256_permute_ps(t2, _MM_SHUFFLE(3, 3, 3, 3));
            x = _mm256_add_ps(x, _mm256_and_ps(t2, mask_low4));

            // Step 4: Add carry from previous 8-element block
            x = _mm256_add_ps(x, v_carry);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Prepare carry for next iteration: broadcast last element of current x
            // Extracting the 7th element (index 3 of high 128-bit lane)
            __m128 high = _mm256_extractf128_ps(x, 1);
            v_carry = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(high, high, _MM_SHUFFLE(3, 3, 3, 3))));
        }
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}