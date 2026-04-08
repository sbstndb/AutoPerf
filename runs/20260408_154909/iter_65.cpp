#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common exponents
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    
    // Fast path for base 2
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Optimization: Handle trailing zeros in exponent by squaring the base.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp is now 1, we are done.
    if (exp == 1) return base;

    uint64_t result = 1;

    // Main Binary Exponentiation Loop (Unrolled for ILP)
    // We process bits of the exponent. If the bit is set, we multiply result by base.
    // We always square the base for the next bit.
    while (exp > 3) {
        // Bit 0
        if (exp & 1) result *= base;
        base *= base;
        
        // Bit 1
        if (exp & 2) result *= base;
        base *= base;
        
        exp >>= 2;
    }

    // Handle remaining bits (exp is now 1, 2, or 3)
    if (exp & 1) result *= base;
    if (exp & 2) {
        base *= base;
        result *= base;
    }

    return result;
}