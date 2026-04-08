#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases: 0^exp, 1^exp, base^0
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Remove trailing zeros from exponent: base^exp = (base^(2^tz))^(exp >> tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero. 
    // Since base > 1, base^(2^tz) overflows 64-bit very quickly (tz < 6).
    while (tz--) {
        base *= base;
    }

    if (exp == 1) return base;

    uint64_t res = 1;

    // Main Binary Exponentiation Loop (LSB to MSB)
    // We unroll slightly and use a technique to minimize branching.
    // The compiler will often optimize the (exp & 1) check into a CMOV 
    // or a sequence that avoids a hard branch.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Optional: Manual unroll for further ILP if exp is large
        if (exp > 1) {
            if (exp & 1) {
                res *= base;
            }
            base *= base;
            exp >>= 1;
        }
    }

    // Final multiplication for the most significant bit
    return res * base;
}