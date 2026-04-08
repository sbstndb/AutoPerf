#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    if (n >= 16) {
        for (; i <= n - 16; i += 16) {
            __m256 x1 = _mm256_loadu_ps(&input[i]);
            __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

            // Intra-lane scan for x1 and x2 in parallel
            // Step 1: shift by 1
            __m256 t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 4));
            __m256 t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 4));
            x1 = _mm256_add_ps(x1, t1);
            x2 = _mm256_add_ps(x2, t2);

            // Step 2: shift by 2
            t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 8));
            t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 8));
            x1 = _mm256_add_ps(x1, t1);
            x2 = _mm256_add_ps(x2, t2);

            // Step 3: Cross-lane (low 128 to high 128)
            __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x00);
            c1 = _mm256_permute_ps(c1, 0xFF);
            x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), c1, 0xF0));

            __m256 c2 = _mm256_permute2f128_ps(x2, x2, 0x00);
            c2 = _mm256_permute_ps(c2, 0xFF);
            x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), c2, 0xF0));

            // Apply global carry to x1
            x1 = _mm256_add_ps(x1, v_carry);

            // Carry for x2 is the last element of x1
            __m128 x1_high = _mm256_extractf128_ps(x1, 1);
            __m128 v_c2_128 = _mm_shuffle_ps(x1_high, x1_high, 0xFF);
            __m256 v_carry_x2 = _mm256_set_m128(v_c2_128, v_c2_128);
            x2 = _mm256_add_ps(x2, v_carry_x2);

            _mm256_storeu_ps(&output[i], x1);
            _mm256_storeu_ps(&output[i + 8], x2);

            // Prepare carry for next iteration from last element of x2
            __m128 x2_high = _mm256_extractf128_ps(x2, 1);
            __m128 v_next_128 = _mm_shuffle_ps(x2_high, x2_high, 0xFF);
            v_carry = _mm256_set_m128(v_next_128, v_next_128);
        }
    }

    // Handle remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t);
        t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t);
        __m256 c = _mm256_permute2f128_ps(x, x, 0x00);
        c = _mm256_permute_ps(c, 0xFF);
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), c, 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        __m128 xh = _mm256_extractf128_ps(x, 1);
        __m128 v_next = _mm_shuffle_ps(xh, xh, 0xFF);
        v_carry = _mm256_set_m128(v_next, v_next);
    }

    // Scalar tail
    float carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}