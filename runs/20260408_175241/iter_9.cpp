#include <cstddef>
#include <immintrin.h>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;

    float running_sum = 0.0f;
    size_t i = 0;

    // Process in blocks of 32 (4 x 8-float AVX registers)
    // This provides enough ILP to hide the latency of the prefix sum logic
    for (; i + 31 < n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(&input[i]);
        __m256 v1 = _mm256_loadu_ps(&input[i + 8]);
        __m256 v2 = _mm256_loadu_ps(&input[i + 16]);
        __m256 v3 = _mm256_loadu_ps(&input[i + 24]);

        auto scan_avx = [](__m256 x) {
            // Intra-lane scan
            __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 4));
            x = _mm256_add_ps(x, t0);
            __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(x), 8));
            x = _mm256_add_ps(x, t1);
            // Cross-lane propagation
            __m256 lane_sum = _mm256_permute2f128_ps(x, x, 0x00);
            lane_sum = _mm256_permute_ps(lane_sum, 0xFF);
            return _mm256_add_ps(x, _mm256_blend_ps(_mm256_setzero_ps(), lane_sum, 0xF0));
        };

        v0 = scan_avx(v0);
        v1 = scan_avx(v1);
        v2 = scan_avx(v2);
        v3 = scan_avx(v3);

        // Apply running sum and carries
        __m256 offset0 = _mm256_set1_ps(running_sum);
        v0 = _mm256_add_ps(v0, offset0);
        
        float c0 = _mm_cvtss_f32(_mm_permute_ps(_mm256_extractf128_ps(v0, 1), 0xFF));
        __m256 offset1 = _mm256_set1_ps(c0);
        v1 = _mm256_add_ps(v1, offset1);

        float c1 = _mm_cvtss_f32(_mm_permute_ps(_mm256_extractf128_ps(v1, 1), 0xFF));
        __m256 offset2 = _mm256_set1_ps(c1);
        v2 = _mm256_add_ps(v2, offset2);

        float c2 = _mm_cvtss_f32(_mm_permute_ps(_mm256_extractf128_ps(v2, 1), 0xFF));
        __m256 offset3 = _mm256_set1_ps(c2);
        v3 = _mm256_add_ps(v3, offset3);

        _mm256_storeu_ps(&output[i], v0);
        _mm256_storeu_ps(&output[i + 8], v1);
        _mm256_storeu_ps(&output[i + 16], v2);
        _mm256_storeu_ps(&output[i + 24], v3);

        running_sum = _mm_cvtss_f32(_mm_permute_ps(_mm256_extractf128_ps(v3, 1), 0xFF));
    }

    // Handle remaining 8-element blocks
    for (; i + 7 < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&input[i]);
        __m256 t0 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 4));
        data = _mm256_add_ps(data, t0);
        __m256 t1 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_castps_si256(data), 8));
        data = _mm256_add_ps(data, t1);
        __m256 lane_sum = _mm256_permute2f128_ps(data, data, 0x00);
        lane_sum = _mm256_permute_ps(lane_sum, 0xFF);
        data = _mm256_add_ps(data, _mm256_blend_ps(_mm256_setzero_ps(), lane_sum, 0xF0));
        
        data = _mm256_add_ps(data, _mm256_set1_ps(running_sum));
        _mm256_storeu_ps(&output[i], data);
        
        running_sum = _mm_cvtss_f32(_mm_permute_ps(_mm256_extractf128_ps(data, 1), 0xFF));
    }

    // Final tail
    for (; i < n; ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }
}