#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Skip trailing zeros in exponent to reduce iterations and seed 'res'
    // This effectively performs res = base^(2^ctz) as the starting point.
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    
    // Pre-square base for the number of trailing zeros
    while (ctz--) {
        base *= base;
    }

    uint64_t res = 1;

    // Standard Binary Exponentiation (Bottom-Up)
    // This allows the CPU to execute 'base *= base' and 'res *= base' 
    // in parallel (ILP) because their dependency chains are separate 
    // within a single iteration.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last remaining bit (MSB is always 1 here)
    return res * base;
}