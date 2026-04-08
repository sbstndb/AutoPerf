#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Remove trailing zeros from exponent to initialize 'result'
    // and reduce the number of iterations.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Squaring the base for the initial trailing zeros
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    if (exp == 0) return result;

    // Determine the highest bit set to avoid unnecessary loop checks
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    // Main loop: Square and Multiply
    // We use a fixed number of iterations to allow the compiler to unroll
    for (int i = 0; i < iterations; ++i) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }
    
    // Final squaring and multiplication for the last bit
    base *= base;
    return result * base;
}