#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Essential fast paths
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
    if (exp == 1) return base;

    // Handle trailing zeros in exponent: base = base^(2^tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done.
    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Determine the highest bit set to bound the loop
    // We already handled the LSB via trailing zeros and checked exp == 1.
    // The loop processes bits from LSB to MSB.
    while (exp > 0) {
        // Use a conditional to avoid branching where possible.
        // The compiler typically optimizes this to a test + cmov or 
        // a simple branch that is highly predictable for many patterns.
        if (exp & 1) {
            res *= base;
        }
        
        // Check if this was the last set bit to avoid an unnecessary final squaring
        exp >>= 1;
        if (exp == 0) break;
        
        base *= base;
    }

    return res;
}