#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common edge case efficiently.
    // __builtin_expect tells the compiler this branch is unlikely.
    if (__builtin_expect(exp == 0, 0)) return 1;

    // Use Count Leading Zeros to find the highest set bit.
    // This defines exactly how many squaring steps are needed.
    // clzll is undefined for 0, but we handled exp == 0 above.
    uint32_t leading_zeros = __builtin_clzll(exp);
    uint32_t iterations = 63 - leading_zeros;

    uint64_t result = 1;

    // We iterate through the bits from least significant to the second-most significant.
    // The most significant bit is handled by the final return to avoid an extra square.
    for (uint32_t i = 0; i < iterations; ++i) {
        // Branchless update: if bit is 0, multiply by 1. If 1, multiply by base.
        // Modern CPUs (especially Intel Ultra) handle this imul/cmov sequence very fast.
        uint64_t bit_set = (exp & 1);
        uint64_t multiplier = bit_set ? base : 1;
        result *= multiplier;
        
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the most significant bit (which is always 1).
    return result * base;
}