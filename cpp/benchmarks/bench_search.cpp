#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

#include "kernel_search.hpp"

static void BM_search(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 0);
  const int key = static_cast<int>(n > 0 ? n - 1 : 0);

  for (auto _ : state) {
    benchmark::DoNotOptimize(autoperf::search_sorted(data, key));
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_search)->RangeMultiplier(2)->Range(1<<10, 1<<24);

BENCHMARK_MAIN();