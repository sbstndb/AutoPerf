#pragma once

#include <cstddef>
#include <vector>

namespace autoperf {

// Linear search in a sorted array. Returns the index of key or -1 if not found.
// Baseline intentionally naive: no early-exit based on sorted property.
int search_sorted(const std::vector<int>& data, int key);

} // namespace autoperf
