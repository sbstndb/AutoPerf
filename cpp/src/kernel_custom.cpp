#include <vector>

// Custom kernel for testing the agnostic system
void custom_operation(const std::vector<float>& input, float* output) {
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        sum += input[i] * input[i];  // Sum of squares
    }
    *output = sum;
}
