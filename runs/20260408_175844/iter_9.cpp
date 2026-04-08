#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    float carry_val = 0.0f;
    __m256 v_carry = _mm256_setzero_ps();

    // Process in blocks of 8 using AVX2
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);

        // Step 1: Shift and add (1-distance)
        // [a, b, c, d | e, f, g, h] -> [0, a, b, c | 0, e, f, g]
        __m256 t0 = _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(x), 
                                        _mm256_permute2f128_si256(_mm256_castps_si256(x), _mm256_castps_si256(x), 0x08), 12));
        // Masking is implicit in alignr for the 128-bit lanes if we handle the cross-lane correctly.
        // However, a simpler Kogge-Stone for AVX2:
        __m256 shift1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, shift1);

        // Step 2: Shift and add (2-distance)
        __m256 shift2 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, shift2);

        // Step 3: Shift and add (4-distance) - Cross 128-bit lane
        __m256 shift4 = _mm256_permute2f128_ps(x, x, 0x20); // Low to High, Zero to Low
        // Mask the low lane to zero because permute2f128 with 0x20 puts low into high and zero into low
        // Actually 0x20: dst[0:127] = 0, dst[128:255] = src1[0:127]
        x = _mm256_add_ps(x, shift4);

        // Step 4: Add the carry from the previous block
        x = _mm256_add_ps(x, v_carry);

        // Store result
        _mm256_storeu_ps(&output[i], x);

        // Broadcast the last element of the current result to all lanes for the next iteration
        // The last element is at index 7.
        __m128 high_lane = _mm256_extractf128_ps(x, 1);
        v_carry = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(high_lane, high_lane, _MM_SHUFFLE(3, 3, 3, 3))));
    }

    // Scalar tail handling
    if (i < n) {
        carry_val = (i == 0) ? 0.0f : output[i-1];
        for (; i < n; ++i) {
            carry_val += input[i];
            output[i] = carry_val;
        }
    }
}