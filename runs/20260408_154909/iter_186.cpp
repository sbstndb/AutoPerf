#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Handle trailing zeros to reduce iterations and potentially exit early
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero. 
    // Manual unrolling or simple loop allows the compiler to use IMUL efficiently.
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }

    if (exp == 1) return base;

    uint64_t res = 1;

    // Binary Exponentiation with reduced branching.
    // We process the exponent bits. The 'if (exp & 1)' is the primary bottleneck.
    // By using a local temporary and a conditional, we encourage the compiler 
    // to use CMOV or better instruction scheduling.
    while (exp > 1) {
        uint64_t side_chain = (exp & 1) ? base : 1;
        res *= side_chain;
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the most significant bit
    return res * base;
}