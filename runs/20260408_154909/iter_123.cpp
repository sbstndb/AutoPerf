#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Skip trailing zeros in the exponent to reduce iterations.
    // This effectively performs: base = base^(2^ctz), res = base.
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    
    // Square base for every trailing zero
    while (ctz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1; // First set bit is already in res

    // Bottom-up binary exponentiation (Square-and-Multiply)
    // This approach allows the CPU to overlap 'base *= base' and 'res *= base'
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}