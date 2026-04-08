#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: Shift and add within 128-bit lanes (distance 1)
        // [a, b, c, d | e, f, g, h] -> [0, a, b, c | 0, e, f, g]
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);

        // Step 2: Shift and add within 128-bit lanes (distance 2)
        // [a, ab, bc, cd | ...] -> [a, ab, abc, abcd | ...]
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);

        // Step 3: Shift and add across 128-bit lanes (distance 4)
        // Broadcast the sum of the lower lane (index 3) to all elements of the upper lane
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20); // Low 128 to High 128, Zero to Low
        t2 = _mm256_permute_ps(t2, 0xFF);               // Broadcast element 3 to all in high lane
        x = _mm256_add_ps(x, t2);

        // Step 4: Add the carry from the previous block
        x = _mm256_add_ps(x, v_carry);

        // Store result
        _mm256_storeu_ps(&output[i], x);

        // Prepare carry for next iteration: broadcast the last element (index 7) of x
        __m256 v_last = _mm256_permute2f128_ps(x, x, 0x11); // High lane to Low lane
        v_carry = _mm256_permute_ps(v_last, 0xFF);          // Broadcast element 7 to all
    }

    // Scalar tail handling
    float carry = (i == 0) ? 0.0f : output[i - 1];
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}