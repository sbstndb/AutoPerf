#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle trailing zeros in exponent to reduce iterations
    // (base^exp) = (base^(2^tz))^ (exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square base for each trailing zero
    while (tz--) {
        base *= base;
    }

    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Standard Binary Exponentiation (LSB to MSB)
    // This allows the CPU to overlap the 'base *= base' chain 
    // with the 'res *= base' chain.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit
    return res * base;
}