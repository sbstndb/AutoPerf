#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for small bases
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Right-to-Left Binary Exponentiation
    // This version is generally faster because 'base' and 'res' 
    // updates can be pipelined more effectively by the CPU.
    
    // Skip trailing zeros in exponent to reduce iterations
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square the base for each trailing zero
    // This is done before entering the loop to simplify logic
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}