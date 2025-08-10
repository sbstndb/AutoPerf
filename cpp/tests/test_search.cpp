#include <gtest/gtest.h>
#include <vector>
#include <numeric>

#include "kernel_search.hpp"

TEST(Search, SmallKnownFound) {
  std::vector<int> data = {1, 3, 5, 7, 9};
  EXPECT_EQ(autoperf::search_sorted(data, 1), 0);
  EXPECT_EQ(autoperf::search_sorted(data, 7), 3);
  EXPECT_EQ(autoperf::search_sorted(data, 9), 4);
}

TEST(Search, SmallKnownNotFound) {
  std::vector<int> data = {2, 4, 6, 8};
  EXPECT_EQ(autoperf::search_sorted(data, 1), -1);
  EXPECT_EQ(autoperf::search_sorted(data, 5), -1);
  EXPECT_EQ(autoperf::search_sorted(data, 10), -1);
}

TEST(Search, RandomMonotonic) {
  const int n = 1000;
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 0);
  // Present keys
  EXPECT_EQ(autoperf::search_sorted(data, 0), 0);
  EXPECT_EQ(autoperf::search_sorted(data, 123), 123);
  EXPECT_EQ(autoperf::search_sorted(data, 999), 999);
  // Absent keys
  EXPECT_EQ(autoperf::search_sorted(data, -1), -1);
  EXPECT_EQ(autoperf::search_sorted(data, 1001), -1);
}


