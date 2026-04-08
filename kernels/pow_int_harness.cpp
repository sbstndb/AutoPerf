// Custom harness for pow_int kernel
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <time.h>
#include <string>

#include "kernel.cpp"

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int run_tests() {
    // Edge cases
    if (pow_int(0, 0) != 1) { fprintf(stderr, "FAIL: 0^0 != 1\n"); return 1; }
    if (pow_int(5, 0) != 1) { fprintf(stderr, "FAIL: 5^0 != 1\n"); return 1; }
    if (pow_int(5, 1) != 5) { fprintf(stderr, "FAIL: 5^1 != 5\n"); return 1; }
    if (pow_int(2, 10) != 1024) { fprintf(stderr, "FAIL: 2^10 != 1024\n"); return 1; }
    if (pow_int(3, 5) != 243) { fprintf(stderr, "FAIL: 3^5 != 243\n"); return 1; }
    if (pow_int(7, 4) != 2401) { fprintf(stderr, "FAIL: 7^4 != 2401\n"); return 1; }
    if (pow_int(2, 20) != 1048576) { fprintf(stderr, "FAIL: 2^20\n"); return 1; }
    if (pow_int(1, 1000000) != 1) { fprintf(stderr, "FAIL: 1^1000000\n"); return 1; }
    return 0;
}

static void run_bench() {
    // Benchmark with a mix of bases and exponents
    const uint64_t bases[] = {2, 3, 4, 5, 7, 11, 13};
    const uint64_t exps[] = {0, 1, 2, 3, 5, 8, 10, 13, 15, 20};
    const int nb = 7, ne = 10;

    volatile uint64_t sink = 0;

    // Warmup
    for (int w = 0; w < 5000; w++)
        for (int b = 0; b < nb; b++)
            for (int e = 0; e < ne; e++)
                sink += pow_int(bases[b], exps[e]);

    // Benchmark
    const int REPS = 50000;
    double start = now_ns();
    for (int r = 0; r < REPS; r++)
        for (int b = 0; b < nb; b++)
            for (int e = 0; e < ne; e++)
                sink += pow_int(bases[b], exps[e]);
    double elapsed = now_ns() - start;

    double ns_per_op = elapsed / (double)(REPS * nb * ne);
    printf("{\"ns_per_op\": %.2f, \"iterations\": %d}\n", ns_per_op, REPS * nb * ne);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--test") return run_tests();
    if (argc > 1 && std::string(argv[1]) == "--bench") { run_bench(); return 0; }
    fprintf(stderr, "Usage: %s --test|--bench\n", argv[0]);
    return 1;
}
