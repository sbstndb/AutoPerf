#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    size_t i = 0;
    __m256 v_carry = _mm256_setzero_ps();

    // Process 16 elements at a time to improve ILP
    for (; i + 15 < n; i += 16) {
        // --- Block 1 (8 elements) ---
        __m256 x1 = _mm256_loadu_ps(&input[i]);
        
        // Kogge-Stone Step 1: Shift by 1 element (4 bytes)
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 4));
        x1 = _mm256_add_ps(x1, t0);
        
        // Kogge-Stone Step 2: Shift by 2 elements (8 bytes)
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x1), 8));
        x1 = _mm256_add_ps(x1, t1);
        
        // Kogge-Stone Step 3: Cross-lane (Shift by 4 elements)
        __m256 t2 = _mm256_permute2f128_ps(x1, x1, 0x20); 
        t2 = _mm256_shuffle_ps(t2, t2, _MM_SHUFFLE(3, 3, 3, 3));
        x1 = _mm256_add_ps(x1, t2);
        
        // Add carry from previous 16-block
        x1 = _mm256_add_ps(x1, v_carry);
        _mm256_storeu_ps(&output[i], x1);

        // Prepare carry for Block 2 from the last element of Block 1
        __m256 v_carry_mid = _mm256_permute2f128_ps(x1, x1, 0x11);
        v_carry_mid = _mm256_shuffle_ps(v_carry_mid, v_carry_mid, _MM_SHUFFLE(3, 3, 3, 3));

        // --- Block 2 (8 elements) ---
        __m256 x2 = _mm256_loadu_ps(&input[i + 8]);
        __m256 t3 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 4));
        x2 = _mm256_add_ps(x2, t3);
        __m256 t4 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x2), 8));
        x2 = _mm256_add_ps(x2, t4);
        __m256 t5 = _mm256_permute2f128_ps(x2, x2, 0x20);
        t5 = _mm256_shuffle_ps(t5, t5, _MM_SHUFFLE(3, 3, 3, 3));
        x2 = _mm256_add_ps(x2, t5);
        x2 = _mm256_add_ps(x2, v_carry_mid);
        _mm256_storeu_ps(&output[i + 8], x2);

        // Final carry for next 16-block loop iteration
        v_carry = _mm256_permute2f128_ps(x2, x2, 0x11);
        v_carry = _mm256_shuffle_ps(v_carry, v_carry, _MM_SHUFFLE(3, 3, 3, 3));
    }

    // Handle remaining 8-element block
    if (i + 7 < n) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
        x = _mm256_add_ps(x, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
        x = _mm256_add_ps(x, t1);
        __m256 t2 = _mm256_permute2f128_ps(x, x, 0x20);
        t2 = _mm256_shuffle_ps(t2, t2, _MM_SHUFFLE(3, 3, 3, 3));
        x = _mm256_add_ps(x, t2);
        x = _mm256_add_ps(x, v_carry);
        _mm256_storeu_ps(&output[i], x);
        
        v_carry = _mm256_permute2f128_ps(x, x, 0x11);
        v_carry = _mm256_shuffle_ps(v_carry, v_carry, _MM_SHUFFLE(3, 3, 3, 3));
        i += 8;
    }

    // Scalar tail
    float scalar_carry = (i == 0) ? 0.0f : _mm_cvtss_f32(_mm256_castps256_ps128(v_carry));
    for (; i < n; ++i) {
        scalar_carry += input[i];
        output[i] = scalar_carry;
    }
}