#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases and small exponents for low latency
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Use trailing zeros to jump-start the 'res' value.
    // This avoids multiplying 'res' by 1 in the early stages.
    int tz = __builtin_ctzll(exp);
    uint64_t res = base;
    for (int i = 0; i < tz; ++i) {
        res *= res;
    }

    // Prepare for the main loop
    uint64_t remaining_exp = exp >> (tz + 1);
    if (remaining_exp == 0) return res;

    uint64_t side = res;
    
    // Calculate how many bits are left to process
    int leading_zeros = __builtin_clzll(remaining_exp);
    int iterations = 64 - leading_zeros;

    // Binary exponentiation loop with reduced branching
    // We use a fixed number of iterations based on the highest set bit
    for (int i = 0; i < iterations; ++i) {
        side *= side;
        if (remaining_exp & 1) {
            res *= side;
        }
        remaining_exp >>= 1;
    }

    return res;
}