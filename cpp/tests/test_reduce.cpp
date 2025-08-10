#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "kernel_reduce.hpp"

TEST(ReduceTest, BasicTest) {
    std::vector<float> input(1024);
    std::iota(input.begin(), input.end(), 1.0f); // Fill with 1.0, 2.0, ..., 1024.0

    float expected_sum = 0.0f;
    for (float val : input) {
        expected_sum += val;
    }

    float actual_sum = 0.0f;
    reduce(input, &actual_sum);

    ASSERT_FLOAT_EQ(expected_sum, actual_sum);
}

TEST(ReduceTest, EmptyTest) {
    std::vector<float> input;
    float expected_sum = 0.0f;
    float actual_sum = 0.0f;
    reduce(input, &actual_sum);
    ASSERT_FLOAT_EQ(expected_sum, actual_sum);
}


