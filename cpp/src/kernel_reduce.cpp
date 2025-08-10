#include "kernel_reduce.hpp"

void reduce(const std::vector<float>& input, float* output) {
    float sum = 0.0f;
    for (float val : input) {
        sum += val;
    }
    *output = sum;
}
