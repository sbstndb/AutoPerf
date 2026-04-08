#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2 x YMM)
    for (; i + 15 < n; i += 16) {
        // Block 1
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        // Intra-lane scan
        __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(1, 0, 0, 0));
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), t0, 0x22));
        t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(1, 0, 0, 0));
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xCC));
        t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(0, 0, 0, 0));
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xF0));
        // Cross-lane
        __m256 c1 = _mm256_permute2f128_ps(x1, x1, 0x00);
        c1 = _mm256_permute_ps(c1, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), c1, 0xF0));
        // Add carry from previous iteration
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);
        // Prepare carry for next block
        v_carry = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));

        // Block 2
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);
        __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(1, 0, 0, 0));
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), t1, 0x22));
        t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(1, 0, 0, 0));
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC));
        t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(0, 0, 0, 0));
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xF0));
        __m256 c2 = _mm256_permute2f128_ps(x2, x2, 0x00);
        c2 = _mm256_permute_ps(c2, _MM_SHUFFLE(3, 3, 3, 3));
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), c2, 0xF0));
        x2 = _mm256_add_ps(x2, v_carry);
        _mm256_storeu_ps(&output[i + 8], x2);
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Process remaining 8-element block
    if (i + 7 < n) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 0, 0));
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), t, 0x22));
        t = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 0, 0));
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), t, 0xCC));
        t = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(0, 0, 0, 0));
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), t, 0xF0));
        __m256 c = _mm256_permute2f128_ps(x, x, 0x00);
        c = _mm256_permute_ps(c, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), c, 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, _MM_SHUFFLE(3, 3, 3, 3));
        i += 8;
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}