#include <cstddef>
#include <immintrin.h>

float reduce(const float* data, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    // Use 8 accumulators to maximize Instruction Level Parallelism (ILP).
    // This helps saturate the FADD units which typically have a latency of 4 cycles.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    // Main loop: Process 64 elements (256 bytes) per iteration.
    for (; i + 63 < n; i += 64) {
        // Prefetch data for future iterations (adjusting distance for L1/L2 latency)
        _mm_prefetch((const char*)(data + i + 128), _MM_HINT_T0);
        
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
        acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps(data + i + 32));
        acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps(data + i + 40));
        acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps(data + i + 48));
        acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps(data + i + 56));
    }

    // Secondary loop: Process 8 elements per iteration for the remainder.
    for (; i + 7 < n; i += 8) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
    }

    // Reduction tree for the 8 accumulators
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc4 = _mm256_add_ps(acc4, acc5);
    acc6 = _mm256_add_ps(acc6, acc7);

    acc0 = _mm256_add_ps(acc0, acc2);
    acc4 = _mm256_add_ps(acc4, acc6);

    acc0 = _mm256_add_ps(acc0, acc4);

    // Horizontal sum of the final 256-bit register
    __m128 low128 = _mm256_castps256_ps128(acc0);
    __m128 high128 = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);

    // Final 128-bit reduction
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(0, 0, 0, 1)));
    sum = _mm_cvtss_f32(sum128);

    // Final scalar cleanup for remaining elements (< 8)
    while (i < n) {
        sum += data[i++];
    }

    return sum;
}