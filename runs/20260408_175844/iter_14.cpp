#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: Shift and add within 128-bit lanes
        // [a, b, c, d | e, f, g, h] -> [a, a+b, b+c, c+d | e, e+f, f+g, g+h]
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);

        // Step 2: Shift and add within 128-bit lanes (distance 2)
        // Result: lane-wise prefix sum
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);

        // Step 3: Propagate lower lane sum to upper lane
        // Extract the 4th element of the lower lane and broadcast it to the upper lane
        __m128 low_lane = _mm256_castps256_ps128(x);
        __m128 low_sum = _mm_permute_ps(low_lane, _MM_SHUFFLE(3, 3, 3, 3));
        __m256 high_prop = _mm256_set_m128(_mm_setzero_ps(), low_sum);
        // Note: _mm256_set_m128(hi, lo) puts hi in upper 128, lo in lower 128.
        // We want the sum of the lower lane added only to the upper lane.
        // Using permute2f128 is often faster:
        __m256 low_to_high = _mm256_permute2f128_ps(x, x, 0x08); // [0, low_lane]
        low_to_high = _mm256_permute_ps(low_to_high, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, low_to_high);

        // Step 4: Add the carry from all previous 8-element blocks
        x = _mm256_add_ps(x, v_carry);

        _mm256_storeu_ps(&output[i], x);

        // Update carry: broadcast the last element of the current 8-element block
        __m128 last_val_vec = _mm256_extractf128_ps(x, 1);
        v_carry = _mm256_broadcastss_ps(_mm_permute_ps(last_val_vec, _MM_SHUFFLE(3, 3, 3, 3)));
    }

    // Scalar tail
    float carry = 0.0f;
    if (i > 0) {
        carry = output[i - 1];
    }

    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}