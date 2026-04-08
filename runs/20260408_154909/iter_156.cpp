#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common/trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    
    // Base 2 is a simple shift. For 64-bit uint, exp >= 64 overflows to 0.
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Pre-square the base for all trailing zeros in the exponent.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done.
    if (exp == 1) return base;

    uint64_t res = 1;

    // Main binary exponentiation loop.
    // We use a technique to minimize branching.
    // The compiler will often optimize the (exp & 1) check, 
    // but we unroll to maximize ILP between 'res' and 'base' updates.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;

        // Unroll level 2
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Check if we can exit early after unrolling
        if (exp <= 1) break;
    }

    // Final bit processing
    if (exp) {
        res *= base;
    }

    return res;
}