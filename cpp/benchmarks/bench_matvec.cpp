#include <benchmark/benchmark.h>
#include "kernel_matvec.hpp"

#include <random>

static void BM_matvec(benchmark::State& state) {
  const std::size_t rows = static_cast<std::size_t>(state.range(0));
  const std::size_t cols = static_cast<std::size_t>(state.range(1));
  std::vector<float> A(rows * cols), x(cols), y(rows, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < rows * cols; ++i) A[i] = dist(rng);
  for (std::size_t j = 0; j < cols; ++j) x[j] = dist(rng);
  for (auto _ : state) {
    std::fill(y.begin(), y.end(), 0.0f);
    autoperf::matvec(A, x, y, rows, cols);
    benchmark::DoNotOptimize(y.data());
  }
  state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(BM_matvec)->Args({256, 1024});

BENCHMARK_MAIN();