#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Handle trailing zeros in exponent: base = base^(2^tz)
    // This reduces the number of iterations in the binary exponentiation loop
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    // Binary Exponentiation (Square and Multiply)
    // Initialize res based on the LSB to skip the first iteration
    uint64_t res = (exp & 1) ? base : 1;
    exp >>= 1;

    // We use __builtin_clzll to find the highest bit and run a fixed number of iterations.
    // This is faster than while(exp) because it's more predictable for the branch predictor.
    int iterations = 63 - __builtin_clzll(exp);
    
    for (int i = 0; i < iterations; ++i) {
        base *= base;
        // Use a temporary to help the compiler use CMOV instead of a branch
        uint64_t side = (exp & 1) ? base : 1;
        res *= side;
        exp >>= 1;
    }
    
    // Final squaring and multiplication for the last bit
    base *= base;
    res *= base;

    return res;
}