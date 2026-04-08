#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time to maximize ILP and hide latency
    if (n >= 16) {
        for (; i <= n - 16; i += 16) {
            __m256 x1 = _mm256_loadu_ps(&input[i]);
            __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

            // Step 1: Intra-lane prefix sum (Shift 1)
            __m256 t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 4));
            __m256 t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 4));
            x1 = _mm256_add_ps(x1, t1);
            x2 = _mm256_add_ps(x2, t2);

            // Step 2: Intra-lane prefix sum (Shift 2)
            t1 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x1), 8));
            t2 = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(x2), 8));
            x1 = _mm256_add_ps(x1, t1);
            x2 = _mm256_add_ps(x2, t2);

            // Step 3: Cross-lane propagation (Low 128 to High 128)
            // Extract element 3 (end of low lane) and broadcast to high lane
            __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x00);
            c1 = _mm256_permute_ps(c1, 0xFF); 
            x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), c1, 0xF0));

            __m256 c2 = _mm256_permute2f128_ps(x2, x2, 0x00);
            c2 = _mm256_permute_ps(c2, 0xFF);
            x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), c2, 0xF0));

            // Step 4: Apply global carry to x1
            x1 = _mm256_add_ps(x1, v_carry);
            
            // Step 5: Extract last element of x1 to use as carry for x2
            __m256 v_carry_x2 = _mm256_permute2f128_ps(x1, x1, 0x11);
            v_carry_x2 = _mm256_permute_ps(v_carry_x2, 0xFF);
            x2 = _mm256_add_ps(x2, v_carry_x2);

            _mm256_storeu_ps(&output[i], x1);
            _mm256_storeu_ps(&output[i + 8], x2);

            // Prepare carry for next iteration (last element of x2)
            v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
            v_carry = _mm256_permute_ps(v_carry, 0xFF);
        }
    }

    // Process remaining 8-element block
    if (i <= n - 8) {
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
        i += 8;
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}