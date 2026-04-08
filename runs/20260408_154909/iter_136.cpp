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

    // Use bit scan to find the highest set bit and reduce loop iterations
    // This removes the need to check 'exp > 1' inside the loop.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;
    
    uint64_t res = 1;

    // Binary Exponentiation (Square and Multiply)
    // We process bits from LSB to the bit before the MSB.
    // The MSB is handled by the final multiplication to save one square/shift.
    for (int i = 0; i < iterations; ++i) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    return res * base;
}