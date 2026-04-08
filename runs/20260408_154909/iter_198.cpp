#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common/simplest cases immediately
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(exp == 1, 1)) return base;
    if (__builtin_expect(base <= 1, 0)) return base;

    // Special case for base 2: use bit shift
    if (__builtin_expect(base == 2, 0)) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Right-to-Left Binary Exponentiation
    // This version is generally faster on modern x86_64 because the 
    // squaring of the base and the multiplication of the result 
    // can be pipelined more effectively by the OOO engine.
    
    // Skip trailing zeros in the exponent to reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for every trailing zero
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    if (__builtin_expect(exp == 0, 1)) return res;

    // Main loop: processes remaining bits
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}