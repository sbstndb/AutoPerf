#include <cstddef>

void axpy(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
