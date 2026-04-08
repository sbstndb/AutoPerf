#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2x unrolling)
    if (n >= 16) {
        for (; i <= n - 16; i += 16) {
            // Load two blocks
            __m256 x0 = _mm256_loadu_ps(&input[i]);
            __m256 x1 = _mm256_loadu_ps(&input[i + 8]);

            // Block 0: Intra-lane scan
            __m256 t0 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x0), 4));
            x0 = _mm256_add_ps(x0, t0);
            t0 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x0), 8));
            x0 = _mm256_add_ps(x0, t0);
            // Cross-lane scan for Block 0
            __m256 c0 = _mm256_permute2f128_ps(x0, x0, 0x20); // Low 128 of x0 into high 128
            c0 = _mm256_shuffle_ps(c0, c0, 0xFF);            // Broadcast index 3 to all
            x0 = _mm256_add_ps(x0, c0);
            x0 = _mm256_add_ps(x0, v_carry);
            _mm256_storeu_ps(&output[i], x0);
            
            // Update carry for Block 1 from Block 0
            v_carry = _mm256_permute2f128_ps(x0, x0, 0x11);
            v_carry = _mm256_shuffle_ps(v_carry, v_carry, 0xFF);

            // Block 1: Intra-lane scan
            __m256 t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 4));
            x1 = _mm256_add_ps(x1, t1);
            t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 8));
            x1 = _mm256_add_ps(x1, t1);
            // Cross-lane scan for Block 1
            __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x20);
            c1 = _mm256_shuffle_ps(c1, c1, 0xFF);
            x1 = _mm256_add_ps(x1, c1);
            x1 = _mm256_add_ps(x1, v_carry);
            _mm256_storeu_ps(&output[i + 8], x1);

            // Update carry for next iteration
            v_carry = _mm256_permute2f128_ps(x1, x1, 0x11);
            v_carry = _mm256_shuffle_ps(v_carry, v_carry, 0xFF);
        }
    }

    // Process remaining 8-element blocks
    for (; i <= (n >= 8 ? n - 8 : 0) && n >= 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t);
        t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t);
        __m256 c = _mm256_permute2f128_ps(x, x, 0x20);
        c = _mm256_shuffle_ps(c, c, 0xFF);
        x = _mm256_add_ps(x, c);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_shuffle_ps(v_carry, v_carry, 0xFF);
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}