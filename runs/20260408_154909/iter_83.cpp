#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Determine the position of the highest set bit.
    // This allows us to run a fixed-count loop, which is much faster
    // than checking 'exp > 1' every iteration.
    int leading_zeros = __builtin_clzll(exp);
    int bits_to_process = 63 - leading_zeros;

    // We process bits from LSB to MSB.
    // To maximize ILP, we use a technique that avoids branching.
    for (int i = 0; i < bits_to_process; ++i) {
        // If the LSB is 1, multiply result by current base.
        // If 0, multiply by 1 (no-op).
        // The ternary here is usually optimized to a CMOV or a mask.
        uint64_t mask = (exp & 1);
        if (mask) {
            result *= base;
        }
        
        base *= base;
        exp >>= 1;
    }

    // The final bit (the original MSB) is always 1 because exp > 0.
    // Multiplying at the end saves one squaring operation.
    return result * base;
}