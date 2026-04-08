#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    float carry = 0.0f;

    if (n >= 8) {
        __m256 v_carry = _mm256_setzero_ps();

        // Process 8 elements at a time
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);

            // Step 1: [a, a+b, b+c, c+d, e, e+f, f+g, g+h]
            __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, t0);

            // Step 2: [a, a+b, a+b+c, a+b+c+d, e, e+f, e+f+g, e+f+g+h]
            __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, t1);

            // Step 3: Cross-lane (low 4 to high 4)
            __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20); // 0x20: low lane of src1 into high lane of dst
            x = _mm256_add_ps(x, t2);

            // Step 4: Add carry from previous block
            x = _mm256_add_ps(x, v_carry);

            _mm256_storeu_ps(&output[i], x);

            // Broadcast the last element of the current result for the next iteration
            // The last element is at index 7.
            __m128 high_lane = _mm256_extractf128_ps(x, 1);
            v_carry = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3))));
        }
        
        // Extract scalar carry for tail processing
        __m128 final_lane = _mm256_extractf128_ps(v_carry, 0);
        carry = _mm_cvtss_f32(final_lane);
    }

    // Scalar tail handling
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}