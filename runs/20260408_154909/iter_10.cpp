#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Special case for base 0
    if (base == 0) return (exp == 0) ? 1 : 0;
    // Special case for exponent 0
    if (exp == 0) return 1;

    // Use __builtin_ctzll to find the first set bit and skip leading zeros.
    // This reduces the number of squaring operations for even exponents.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for the initial trailing zeros
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Process remaining bits of the exponent
    while (exp > 0) {
        base *= base;
        
        // If the bit is set, multiply result. 
        // Using a temporary and a conditional ensures the compiler can 
        // use a CMOV or simple arithmetic to avoid a branch.
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;
        
        exp >>= 1;
    }

    return result;
}