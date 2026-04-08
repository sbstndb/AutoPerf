#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time (2 x 8-element YMM registers)
    for (; i + 15 < n; i += 16) {
        // Load first 8 elements
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        
        // Intra-lane scan (Kogge-Stone)
        // Shift 1: [0, a, b, c]
        __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(2, 1, 0, 3));
        t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
        x1 = _mm256_add_ps(x1, t0);
        
        // Shift 2: [0, 0, a, a+b]
        __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x1, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
        x1 = _mm256_add_ps(x1, t1);
        
        // Cross-lane scan: Add sum of low 128-bit to high 128-bit
        __m256 t2 = _mm256_permute2f128_ps(x1, x1, 0x20); // Low lane to high lane
        t2 = _mm256_permute_ps(t2, 0xFF); // Broadcast last element of low lane
        x1 = _mm256_add_ps(x1, _mm256_blend_ps(_mm256_setzero_ps(), t2, 0xF0));
        
        // Add carry from previous iterations
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);
        
        // Update carry for the next 8 elements
        v_carry = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);

        // Load next 8 elements
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);
        
        // Repeat scan for second block
        __m256 t3 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(2, 1, 0, 3));
        t3 = _mm256_blend_ps(_mm256_setzero_ps(), t3, 0xEE);
        x2 = _mm256_add_ps(x2, t3);
        
        __m256 t4 = _mm256_shuffle_ps(_mm256_setzero_ps(), x2, _MM_SHUFFLE(1, 0, 3, 2));
        t4 = _mm256_blend_ps(_mm256_setzero_ps(), t4, 0xCC);
        x2 = _mm256_add_ps(x2, t4);
        
        __m256 t5 = _mm256_permute2f128_ps(x2, x2, 0x20);
        t5 = _mm256_permute_ps(t5, 0xFF);
        x2 = _mm256_add_ps(x2, _mm256_blend_ps(_mm256_setzero_ps(), t5, 0xF0));
        
        x2 = _mm256_add_ps(x2, v_carry);
        _mm256_storeu_ps(&output[i + 8], x2);
        
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Process remaining 8-element blocks
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(2, 1, 0, 3));
        t0 = _mm256_blend_ps(_mm256_setzero_ps(), t0, 0xEE);
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_shuffle_ps(_mm256_setzero_ps(), x, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xCC);
        x = _mm256_add_ps(x, t1);
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20);
        t2 = _mm256_permute_ps(t2, 0xFF);
        x = _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), t2, 0xF0));
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_permute_ps(v_carry, 0xFF);
    }

    // Final scalar tail
    float carry = (i == 0) ? 0.0f : _mm256_cvtss_f32(v_carry);
    for (; i < n; ++i) {
        carry += input[i];
        output[i] = carry;
    }
}