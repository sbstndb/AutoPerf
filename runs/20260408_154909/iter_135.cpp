#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special cases for base 0, 1, 2
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    uint64_t res = 1;

    // Binary Exponentiation (Square and Multiply)
    // We use __builtin_clzll to find the highest set bit.
    // This avoids unnecessary iterations and helps the optimizer.
    int leading_zeros = __builtin_clzll(exp);
    int bit_width = 64 - leading_zeros;

    // Process all bits except the most significant one
    for (int i = 0; i < bit_width - 1; ++i) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The most significant bit is always 1 (since exp > 0 and handled by loop logic)
    return res * base;
}