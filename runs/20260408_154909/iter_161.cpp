#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Handle trailing zeros in exponent: base^(n * 2^tz) = (base^(2^tz))^n
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Binary exponentiation (Square and Multiply)
    // We use a bit-scan to limit iterations and avoid unnecessary work.
    // The loop is structured to encourage the compiler to use IMUL and 
    // minimize branch misprediction penalties.
    int bits = 64 - __builtin_clzll(exp);
    
    for (int i = 0; i < bits; ++i) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    return res;
}