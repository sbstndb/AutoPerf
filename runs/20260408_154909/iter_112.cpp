#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common edge cases
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(base <= 1, 0)) return base;
    
    // Handle small exponents immediately
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros: base = base^(2^tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Unrolled squaring for trailing zeros to maximize ILP
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    uint64_t result = base;
    exp >>= 1;
    if (exp == 0) return result;

    // Main Binary Exponentiation loop unrolled by 2
    // This allows the squaring of 'base' to stay ahead of the 'result' update
    while (exp >= 2) {
        // Bit 0
        base *= base;
        if (exp & 1) result *= base;
        
        // Bit 1
        base *= base;
        if (exp & 2) result *= base;
        
        exp >>= 2;
    }

    // Final bit if necessary
    if (exp & 1) {
        base *= base;
        result *= base;
    }

    return result;
}