// Custom harness for prefix_sum kernel
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <time.h>
#include <string>
#include <vector>

#include "kernel.cpp"

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int run_tests() {
    // Test 1: small known case
    {
        const float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        const float expected[] = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
        float out[5] = {};
        prefix_sum(in, out, 5);
        for (int i = 0; i < 5; i++) {
            if (std::fabs(out[i] - expected[i]) > 1e-4f) {
                fprintf(stderr, "FAIL small: out[%d] = %f, expected %f\n", i, out[i], expected[i]);
                return 1;
            }
        }
    }

    // Test 2: single element
    {
        const float in[] = {42.0f};
        float out[1] = {};
        prefix_sum(in, out, 1);
        if (std::fabs(out[0] - 42.0f) > 1e-6f) {
            fprintf(stderr, "FAIL single: got %f\n", out[0]);
            return 1;
        }
    }

    // Test 3: empty
    {
        float out[1] = {999.0f};
        prefix_sum(nullptr, out, 0);
        // should not crash
    }

    // Test 4: larger random — verify last element equals total sum
    {
        const size_t N = 10000;
        std::vector<float> in(N), out(N);
        float total = 0.0f;
        for (size_t i = 0; i < N; i++) {
            in[i] = (float)(i % 17) * 0.1f;
            total += in[i];
        }
        prefix_sum(in.data(), out.data(), N);
        // Check monotonicity and final value
        for (size_t i = 1; i < N; i++) {
            if (out[i] < out[i-1] - 1e-2f) {
                fprintf(stderr, "FAIL monotonicity at %zu: %f < %f\n", i, out[i], out[i-1]);
                return 1;
            }
        }
        if (std::fabs(out[N-1] - total) > 1.0f) {
            fprintf(stderr, "FAIL total: got %f, expected %f\n", out[N-1], total);
            return 1;
        }
    }

    return 0;
}

static void run_bench() {
    const size_t sizes[] = {1024, 4096, 16384, 65536};
    const int nsizes = 4;

    // Use the middle size for the main benchmark
    const size_t N = 16384;
    std::vector<float> input(N), output(N);
    for (size_t i = 0; i < N; i++) input[i] = (float)(i % 100) * 0.01f;

    volatile float sink = 0.0f;

    // Warmup
    for (int w = 0; w < 2000; w++) {
        prefix_sum(input.data(), output.data(), N);
        sink += output[N-1];
    }

    // Benchmark
    const int REPS = 20000;
    double start = now_ns();
    for (int r = 0; r < REPS; r++) {
        prefix_sum(input.data(), output.data(), N);
        sink += output[N-1];
    }
    double elapsed = now_ns() - start;

    double ns_per_op = elapsed / (double)REPS;
    printf("{\"ns_per_op\": %.2f, \"iterations\": %d, \"N\": %zu}\n", ns_per_op, REPS, N);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--test") return run_tests();
    if (argc > 1 && std::string(argv[1]) == "--bench") { run_bench(); return 0; }
    fprintf(stderr, "Usage: %s --test|--bench\n", argv[0]);
    return 1;
}
