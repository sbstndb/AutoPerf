#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents and bases
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(base == 0, 0)) return 0;
    if (exp == 1) return base;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Handle trailing zeros in exponent: result = (base^(2^tz))^exp_odd
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Unrolled squaring for trailing zeros to increase ILP
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    uint64_t result = base;
    exp >>= 1;
    if (exp == 0) return result;

    // Determine the highest bit set to limit iterations
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    // Main Square and Multiply loop
    // We use a technique to minimize branching. 
    // Modern CPUs handle the 'if' well if the exponent bits are biased,
    // but for random exponents, a multiplier-selection is more stable.
    for (int i = 0; i < iterations; ++i) {
        base *= base;
        uint64_t side = (exp & 1) ? base : 1;
        result *= side;
        exp >>= 1;
    }

    return result;
}