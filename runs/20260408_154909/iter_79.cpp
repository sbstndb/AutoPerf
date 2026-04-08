#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases for 0 and 1
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Skip trailing zeros in the exponent to reduce iterations.
    // This effectively performs: result = base^(2^ctz); base = base^(2^ctz)
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    
    // Square the base for each trailing zero bit
    while (ctz--) {
        base *= base;
    }

    // Initialize result with the base corresponding to the first set bit
    uint64_t result = base;
    exp >>= 1;

    // Standard Exponentiation by Squaring (Binary Exponentiation)
    // We process only the remaining bits.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}