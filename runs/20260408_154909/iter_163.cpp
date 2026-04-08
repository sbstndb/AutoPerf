#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Use __builtin_ctzll to handle trailing zeros in exponent (base^(2^k))
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square the base for each trailing zero
    while (tz--) {
        base *= base;
    }

    if (exp == 1) return base;

    // Binary Exponentiation (Square-and-Multiply)
    // We start from the highest set bit to avoid an extra 'res = 1' multiplication
    uint64_t res = base;
    int shift = 63 - __builtin_clzll(exp);
    
    // Process bits from second-highest to lowest
    while (shift--) {
        res *= res;
        if ((exp >> shift) & 1) {
            res *= base;
        }
    }

    return res;
}