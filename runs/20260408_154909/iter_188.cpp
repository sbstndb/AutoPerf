#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Fast path for small exponents to avoid ctz/shift overhead
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros: (base^exp) = (base^(2^tz))^(exp >> tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square base for each trailing zero. 
    // Unrolled slightly to help ILP.
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) {
        base *= base;
    }

    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Optimized Binary Exponentiation
    // We use a technique to minimize branching.
    // The compiler will often optimize the 'if (exp & 1)' into a CMOV
    // but we structure it to ensure the 'base' squaring chain 
    // is independent of the 'res' chain as much as possible.
    while (exp > 1) {
        uint64_t side = (exp & 1) ? base : 1;
        res *= side;
        base *= base;
        exp >>= 1;
    }
    
    return res * base;
}