#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 32 elements at a time to maximize ILP
    if (n >= 32) {
        for (; i <= n - 32; i += 32) {
            __m256 x0 = _mm256_loadu_ps(&input[i]);
            __m256 x1 = _mm256_loadu_ps(&input[i + 8]);
            __m256 x2 = _mm256_loadu_ps(&input[i + 16]);
            __m256 x3 = _mm256_loadu_ps(&input[i + 24]);

            // Step 1: Intra-lane prefix sum (shift 4, add, shift 8, add)
            auto intra_ps = [](__m256 x) {
                __m256 t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4));
                x = _mm256_add_ps(x, t);
                t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8));
                x = _mm256_add_ps(x, t);
                // Cross-lane: propagate high element of low 128 to all elements of high 128
                __m256 c = _mm256_permute2f128_ps(x, x, 0x00);
                c = _mm256_permute_ps(c, 0xFF);
                return _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), c, 0xF0));
            };

            x0 = intra_ps(x0);
            x1 = intra_ps(x1);
            x2 = intra_ps(x2);
            x3 = intra_ps(x3);

            // Step 2: Inter-block carry propagation
            // x0 uses v_carry
            x0 = _mm256_add_ps(x0, v_carry);

            // x1 uses last element of x0
            __m256 c0 = _mm256_permute2f128_ps(x0, x0, 0x11);
            c0 = _mm256_permute_ps(c0, 0xFF);
            x1 = _mm256_add_ps(x1, c0);

            // x2 uses last element of x1
            __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x11);
            c1 = _mm256_permute_ps(c1, 0xFF);
            x2 = _mm256_add_ps(x2, c1);

            // x3 uses last element of x2
            __m256 c2 = _mm256_permute2f128_ps(x2, x2, 0x11);
            c2 = _mm256_permute_ps(c2, 0xFF);
            x3 = _mm256_add_ps(x3, c2);

            _mm256_storeu_ps(&output[i], x0);
            _mm256_storeu_ps(&output[i + 8], x1);
            _mm256_storeu_ps(&output[i + 16], x2);
            _mm256_storeu_ps(&output[i + 24], x3);

            // Update global carry for next loop
            v_carry = _mm256_permute2f128_ps(x3, x3, 0x11);
            v_carry = _mm256_permute_ps(v_carry, 0xFF);
        }
    }

    // Process remaining 8-element blocks
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
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Scalar tail
    float carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}