#include <benchmark/benchmark.h>
#include "kernel_matmul.hpp"

#include <random>

static void BM_matmul(benchmark::State& state) {
  const std::size_t M = static_cast<std::size_t>(state.range(0));
  const std::size_t N = static_cast<std::size_t>(state.range(1));
  const std::size_t K = static_cast<std::size_t>(state.range(2));
  
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  for (std::size_t i = 0; i < M * K; ++i) A[i] = dist(rng);
  for (std::size_t i = 0; i < K * N; ++i) B[i] = dist(rng);
  
  for (auto _ : state) {
    std::fill(C.begin(), C.end(), 0.0f);
    autoperf::matmul(A, B, C, M, N, K);
    benchmark::DoNotOptimize(C.data());
  }
  state.SetItemsProcessed(state.iterations() * M * N * K);
}

BENCHMARK(BM_matmul)->Args({128, 128, 128});

BENCHMARK_MAIN();