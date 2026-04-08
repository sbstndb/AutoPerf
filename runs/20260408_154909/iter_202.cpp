#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
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

    // Right-to-Left Binary Exponentiation (Square-and-Multiply)
    // This version is generally faster because it allows for better ILP
    // and we can skip zero bits efficiently.
    uint64_t res = 1;

    // Skip trailing zeros in exponent to initialize 'res' and 'base'
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for the initial trailing zeros
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    
    // First bit is now guaranteed to be 1
    res = base;
    exp >>= 1;

    // Process remaining bits
    while (__builtin_expect(exp > 0, 1)) {
        base *= base; // Square the base
        if (exp & 1) {
            res *= base; // Multiply result if bit is set
        }
        exp >>= 1;
        
        // Optimization: if remaining exp is even, we can skip multiple squarings
        // using ctzll, but for 64-bit ints, a simple shift is often lower latency.
        if (__builtin_expect(exp > 1 && (exp & 1) == 0, 0)) {
            int skip = __builtin_ctzll(exp);
            for (int i = 0; i < skip; ++i) {
                base *= base;
            }
            exp >>= skip;
            res *= base;
            exp >>= 1;
        }
    }

    return res;
}