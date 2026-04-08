#include <cstddef>

void prefix_sum(const float* input, float* output, size_t n) {
    if (n == 0) return;
    output[0] = input[0];
    for (size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}
