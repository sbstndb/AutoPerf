#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>
#include "kernel_reduce.hpp"

static void BM_reduce(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input(size);
    std::iota(input.begin(), input.end(), 1.0f);
    float output;

    for (auto _ : state) {
        reduce(input, &output);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
}

BENCHMARK(BM_reduce)->RangeMultiplier(2)->Range(1<<10, 1<<24);

BENCHMARK_MAIN();