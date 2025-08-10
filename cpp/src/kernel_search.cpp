#include "kernel_search.hpp"

namespace autoperf {

int search_sorted(const std::vector<int>& data, int key) {
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i] == key) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

} // namespace autoperf
