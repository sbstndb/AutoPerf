#include "kernel_reduce.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  constexpr std::size_t size = 1024;
  std::vector<float> input(size);
  std::iota(input.begin(), input.end(), 1.0f);
  float output;

  auto start = std::chrono::high_resolution_clock::now();
  reduce(input, &output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = end - start;
  std::cout << "reduce completed in " << ms.count() << " ms\n";

  std::cout << "sum: " << output << "\n";
  return 0;
}
