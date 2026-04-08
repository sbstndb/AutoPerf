#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp=0 -> 1, base=0 -> 0, base=1 -> 1
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2 (2^64 wraps to 0 in 64-bit unsigned)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Use __builtin_ctzll to skip trailing zeros in exponent.
    // This effectively transforms base^exp into (base^(2^tz))^(exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Fast squaring for trailing zeros
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;
    
    if (exp == 0) return res;

    // Binary Exponentiation with manual unrolling for ILP
    // We calculate squares of the base and conditionally multiply into res.
    // The compiler will use CMOV for the ternary to avoid branch mispredicts.
    while (exp >= 4) {
        uint64_t b2 = base * base;
        uint64_t b4 = b2 * b2;
        uint64_t b8 = b4 * b4;
        
        if (exp & 1) res *= b2;
        if (exp & 2) res *= b4;
        if (exp & 4) res *= b8;
        
        base = b8 * b8;
        exp >>= 3; // We processed 3 bits (2, 4, 8) but base is now ready for bit 16
    }

    // Final bits
    while (exp > 0) {
        base *= base;
        if (exp & 1) res *= base;
        exp >>= 1;
    }

    return res;
}