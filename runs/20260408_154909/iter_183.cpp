#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle trailing zeros in exponent to reduce iterations
    // (base^exp) = (base^(2^tz))^ (exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square base for each trailing zero.
    // Using a simple loop; for uint64_t, tz is rarely large.
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }

    if (exp == 1) return base;

    // We use the MSB-to-LSB approach or a highly optimized LSB-to-MSB.
    // To maximize ILP and eliminate branches, we pre-calculate the loop count.
    uint64_t res = 1;
    
    // Number of bits to process after the LSB
    int bits = 63 - __builtin_clzll(exp);

    // Standard Binary Exponentiation (LSB to MSB)
    // We remove the branch inside the loop using a conditional multiplier.
    // This allows the CPU to schedule multiplications more aggressively.
    while (bits--) {
        uint64_t side_effect = (exp & 1);
        // If bit is 0, multiplier is 1. If bit is 1, multiplier is base.
        // This is often faster than a branch on modern out-of-order CPUs.
        res = (side_effect ? res * base : res);
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit (which is always 1)
    return res * base;
}