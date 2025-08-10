#include "kernel_search.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  constexpr std::size_t n = 1000;
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 0);
  const int key = static_cast<int>(n > 0 ? n - 1 : 0);

  auto start = std::chrono::high_resolution_clock::now();
  int result = autoperf::search_sorted(data, key);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = end - start;
  std::cout << "search_sorted completed in " << ms.count() << " ms\n";

  std::cout << "result: " << result << "\n";
  return 0;
}
