#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle trailing zeros in exponent: (base^exp) = (base^(2^tz))^(exp >> tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Squaring the base for trailing zeros. 
    // Since base >= 2, base^(2^tz) overflows 64-bit very quickly (tz > 5).
    // A simple unrolled loop is more efficient than a generic while loop.
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }

    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Optimized Binary Exponentiation
    // We use a branchless approach to update 'res' to avoid mispredictions.
    // The compiler will typically optimize 'res * (exp & 1 ? base : 1)' 
    // into a CMOV or a simple multiplication sequence.
    while (exp > 1) {
        uint64_t side_effect = (exp & 1) ? base : 1;
        res *= side_effect;
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit (MSB is always 1 here)
    return res * base;
}