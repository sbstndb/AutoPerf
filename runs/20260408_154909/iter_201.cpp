#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for small exponents (0, 1, 2)
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Right-to-Left Binary Exponentiation
    // This approach allows better ILP as 'res' and 'base' updates 
    // can be pipelined by the CPU.
    uint64_t res = 1;

    // Skip trailing zeros in exponent to reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for the skipped trailing zeros
    // Using a small loop or unrolling for squaring
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    // Process the remaining bits
    // We know the LSB of exp is now 1
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // The last bit is always 1, so multiply res by the final base
    return res * base;
}