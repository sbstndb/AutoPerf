#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2 x 8-float YMM registers)
    for (; i + 15 < n; i += 16) {
        // Load two blocks
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);

        // Scan Block 1 (Intra-register)
        __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(2, 1, 0, 3));
        t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
        x1 = _mm256_add_ps(x1, t0);
        __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
        x1 = _mm256_add_ps(x1, t1);
        __m256 t2 = _mm256_permute2f128_ps(x1, x1, 0x20);
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(t2, 0xFF), 0xF0));
        
        // Apply carry to Block 1 and store
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);
        
        // Extract carry for Block 2 from Block 1
        v_carry = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);

        // Scan Block 2 (Intra-register)
        __m256 t3 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(2, 1, 0, 3));
        t3 = _mm256_blend_ps(_mm256_setzero_ps(), t3, 0xEE);
        x2 = _mm256_add_ps(x2, t3);
        __m256 t4 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(1, 0, 3, 2));
        t4 = _mm256_blend_ps(_mm256_setzero_ps(), t4, 0xCC);
        x2 = _mm256_add_ps(x2, t4);
        __m256 t5 = _mm256_permute2f128_ps(x2, x2, 0x20);
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(t5, 0xFF), 0xF0));

        // Apply carry to Block 2 and store
        x2 = _mm256_add_ps(x2, v_carry);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Extract carry for next iteration
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Process remaining 8-element block
    if (i + 7 < n) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(2, 1, 0, 3));
        t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
        x = _mm256_add_ps(x, t1);
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20);
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), _mm256_permute_ps(t2, 0xFF), 0xF0));
        
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
        i += 8;
    }

    // Scalar tail
    float carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}