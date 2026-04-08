#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases: 0^0 is 1, 1^exp is 1, 0^exp (exp>0) is 0
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Handle trailing zeros in exponent: (base^exp) = (base^(2^tz))^(exp >> tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero.
    // Modern CPUs handle this dependency chain well.
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done.
    if (exp == 1) return base;

    uint64_t res = 1;

    // Determine the number of bits remaining to process.
    // We start from the LSB and move towards the MSB.
    // Using a fixed number of iterations based on the leading zero count
    // helps the CPU's branch predictor and loop unroller.
    int bits = 64 - __builtin_clzll(exp);

    // Standard Binary Exponentiation (Square and Multiply)
    // We use a branchless approach for the multiplication to maintain 
    // a consistent execution pipeline.
    for (int i = 0; i < bits - 1; ++i) {
        uint64_t mask = -(exp & 1);
        res *= (base & mask) | (1ULL & ~mask); // res *= (exp & 1) ? base : 1
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the most significant bit (which is always 1)
    return res * base;
}