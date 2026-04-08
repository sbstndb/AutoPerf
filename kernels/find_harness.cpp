// Custom harness for find kernel — worst case (target at last position)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <string>

#include "kernel.cpp"

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int run_tests() {
    // Test 1: find at end
    {
        const int N = 16;
        int data[16] = {};
        data[15] = 1;
        int idx = find(data, N, 1);
        if (idx != 15) { fprintf(stderr, "FAIL: find at end, got %d\n", idx); return 1; }
    }
    // Test 2: find at start
    {
        int data[] = {42, 0, 0, 0};
        int idx = find(data, 4, 42);
        if (idx != 0) { fprintf(stderr, "FAIL: find at start, got %d\n", idx); return 1; }
    }
    // Test 3: not found
    {
        int data[] = {1, 2, 3, 4};
        int idx = find(data, 4, 99);
        if (idx != -1) { fprintf(stderr, "FAIL: not found, got %d\n", idx); return 1; }
    }
    // Test 4: larger array
    {
        const int N = 4096;
        int* data = (int*)calloc(N, sizeof(int));
        data[N - 1] = 1;
        int idx = find(data, N, 1);
        if (idx != N - 1) { fprintf(stderr, "FAIL: large find, got %d expected %d\n", idx, N-1); free(data); return 1; }
        free(data);
    }
    // Test 5: single element
    {
        int data[] = {7};
        if (find(data, 1, 7) != 0) { fprintf(stderr, "FAIL: single found\n"); return 1; }
        if (find(data, 1, 8) != -1) { fprintf(stderr, "FAIL: single not found\n"); return 1; }
    }
    return 0;
}

static void run_bench() {
    const int N = 4096;
    int* data = (int*)aligned_alloc(64, N * sizeof(int));
    memset(data, 0, N * sizeof(int));
    data[N - 1] = 1;  // worst case: target at last position

    volatile int sink = 0;

    // Warmup
    for (int w = 0; w < 5000; w++)
        sink += find(data, N, 1);

    const int REPS = 100000;
    double start = now_ns();
    for (int r = 0; r < REPS; r++)
        sink += find(data, N, 1);
    double elapsed = now_ns() - start;

    double ns_per_op = elapsed / (double)REPS;
    printf("{\"ns_per_op\": %.2f, \"iterations\": %d, \"N\": %d}\n", ns_per_op, REPS, N);
    free(data);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--test") return run_tests();
    if (argc > 1 && std::string(argv[1]) == "--bench") { run_bench(); return 0; }
    fprintf(stderr, "Usage: %s --test|--bench\n", argv[0]);
    return 1;
}
