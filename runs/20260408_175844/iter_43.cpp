#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time
    if (n >= 16) {
        for (; i <= n - 16; i += 16) {
            __m256 x1 = _mm256_loadu_ps(&input[i]);
            __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

            // Internal prefix sum for x1
            __m256 t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 4));
            x1 = _mm256_add_ps(x1, t1);
            t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 8));
            x1 = _mm256_add_ps(x1, t1);
            __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x00);
            c1 = _mm256_permute_ps(c1, 0xFF);
            x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), c1, 0xF0));

            // Internal prefix sum for x2
            __m256 t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 4));
            x2 = _mm256_add_ps(x2, t2);
            t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 8));
            x2 = _mm256_add_ps(x2, t2);
            __m256 c2 = _mm256_permute2f128_ps(x2, x2, 0x00);
            c2 = _mm256_permute_ps(c2, 0xFF);
            x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), c2, 0xF0));

            // Apply global carry to both
            x1 = _mm256_add_ps(x1, v_carry);
            
            // Carry for x2 is (global carry + last element of x1 before global carry)
            // which is simply the last element of the updated x1.
            __m256 v_carry_x2 = _mm256_permute2f128_ps(x1, x1, 0x11);
            v_carry_x2 = _mm256_permute_ps(v_carry_x2, 0xFF);
            x2 = _mm256_add_ps(x2, v_carry_x2);

            _mm256_storeu_ps(&output[i], x1);
            _mm256_storeu_ps(&output[i + 8], x2);

            // Extract carry for next iteration from last element of x2
            v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
            v_carry = _mm256_permute_ps(v_carry, 0xFF);
        }
    }

    // Process remaining 8-element blocks
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        t0 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t0);
        __m256 c = _mm256_permute2f128_ps(x, x, 0x00);
        c = _mm256_permute_ps(c, 0xFF);
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), c, 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute2f128_ps(v_carry, v_carry, 0x11)));
    if (i > 0) {
        // Correctly extract the very last element from the v_carry vector
        scalar_carry = output[i-1];
    }

    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}