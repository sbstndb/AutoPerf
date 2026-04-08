#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Remove trailing zeros from exp to reduce iterations.
    // base^(base^(2^tz)) is handled by squaring base tz times.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done.
    if (exp == 1) return base;

    // Initialize res with base and move to the next bit.
    // This saves one 'res = 1' multiplication.
    uint64_t res = base;
    exp >>= 1;

    // Main binary exponentiation loop.
    // We use a pattern that encourages the compiler to use IMUL and CMOV
    // to avoid branch mispredictions on the bits of 'exp'.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        
        // Manual unroll to increase ILP if exp is large
        if (exp == 0) break;
        
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}