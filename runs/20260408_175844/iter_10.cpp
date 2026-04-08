#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    if (n >= 8) {
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: Internal scan within 4-element lanes
            // [a, b, c, d] -> [a, a+b, b+c, c+d]
            __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(2, 1, 0, 3));
            t0 = _mm256_and_ps(t0, _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, 0, -1, -1, -1, 0)));
            x = _mm256_add_ps(x, t0);

            // [a, a+b, b+c, c+d] -> [a, a+b, a+b+c, a+b+c+d]
            __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 3, 2));
            t1 = _mm256_and_ps(t1, _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, 0, 0, -1, -1, 0, 0)));
            x = _mm256_add_ps(x, t1);

            // Step 2: Propagate low 128-bit sum to high 128-bit lane
            // Extract the sum of the first 4 elements (index 3) and broadcast to high lane
            __m256 t2 = _mm256_permute2f128_ps(x, x, 0x00); // Low lane to both
            t2 = _mm256_permute_ps(t2, _MM_SHUFFLE(3, 3, 3, 3));
            // Mask out the low lane so we only add to the high lane
            t2 = _mm256_and_ps(t2, _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0)));
            x = _mm256_add_ps(x, t2);

            // Step 3: Add carry from previous 8-element block
            x = _mm256_add_ps(x, v_carry);

            // Store result
            _mm256_storeu_ps(&output[i], x);

            // Step 4: Prepare carry for next iteration
            // Broadcast the last element (index 7) of the current result
            __m256 v_last = _mm256_permute2f128_ps(x, x, 0x11); // High lane to both
            v_carry = _mm256_permute_ps(v_last, _MM_SHUFFLE(3, 3, 3, 3));
        }
    }

    // Scalar tail
    float carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}