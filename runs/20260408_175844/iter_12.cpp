#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time to hide latency of carry propagation
    if (n >= 16) {
        for (; i <= n - 16; i += 16) {
            // Block 1
            __m256 x1 = _mm256_loadu_ps(&input[i]);
            __m256 t0 = _mm256_permute_ps(x1, _MM_SHUFFLE(2, 1, 0, 0));
            t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
            x1 = _mm256_add_ps(x1, t0);
            __m256 t1 = _mm256_permute_ps(x1, _MM_SHUFFLE(1, 0, 0, 0));
            t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
            x1 = _mm256_add_ps(x1, t1);
            __m256 t2 = _mm256_permute2f128_ps(x1, x1, 0x20);
            t2 = _mm256_permute_ps(t2, _MM_SHUFFLE(3, 3, 3, 3));
            x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), t2, 0xF0));
            
            x1 = _mm256_add_ps(x1, v_carry);
            _mm256_storeu_ps(&output[i], x1);
            v_carry = _mm256_permute2f128_ps(x1, x1, 0x11);
            v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));

            // Block 2
            __m256 x2 = _mm256_loadu_ps(&input[i + 8]);
            __m256 t3 = _mm256_permute_ps(x2, _MM_SHUFFLE(2, 1, 0, 0));
            t3 = _mm256_blend_ps(_mm256_setzero_ps(), t3, 0xEE);
            x2 = _mm256_add_ps(x2, t3);
            __m256 t4 = _mm256_permute_ps(x2, _MM_SHUFFLE(1, 0, 0, 0));
            t4 = _mm256_blend_ps(_mm256_setzero_ps(), t4, 0xCC);
            x2 = _mm256_add_ps(x2, t4);
            __m256 t5 = _mm256_permute2f128_ps(x2, x2, 0x20);
            t5 = _mm256_permute_ps(t5, _MM_SHUFFLE(3, 3, 3, 3));
            x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), t5, 0xF0));

            x2 = _mm256_add_ps(x2, v_carry);
            _mm256_storeu_ps(&output[i + 8], x2);
            v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
            v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
        }
    }

    // Process remaining 8-element blocks
    for (; i <= (n >= 8 ? n - 8 : 0) && n >= 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 0));
        t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 0, 0));
        t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
        x = _mm256_add_ps(x, t1);
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20);
        t2 = _mm256_permute_ps(t2, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), t2, 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}