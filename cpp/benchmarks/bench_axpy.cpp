#include <benchmark/benchmark.h>
#include "kernel_axpy.hpp"

#include <random>

static void BM_axpy(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  std::vector<float> x(n), y(n);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }

  for (auto _ : state) {
    std::vector<float> y_copy = y;
    autoperf::axpy(2.0f, x, y_copy);
    benchmark::DoNotOptimize(y_copy.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_axpy)->Arg(1 << 20);

BENCHMARK_MAIN();


