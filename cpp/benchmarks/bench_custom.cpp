#include <benchmark/benchmark.h>
#include "kernel_custom.hpp"

#include <random>

static void BM_custom(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  std::vector<float> input(n);
  float output = 0.0f;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) input[i] = dist(rng);
  for (auto _ : state) {
    custom_operation(input, &output);
    benchmark::DoNotOptimize(&output);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_custom)->Arg(100000);

BENCHMARK_MAIN();
