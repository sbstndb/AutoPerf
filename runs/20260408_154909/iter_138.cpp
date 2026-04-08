#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents
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

    // Use Bit Scan Reverse to find the highest set bit
    // This allows us to skip all leading zeros in the exponent
    int leading_zeros = __builtin_clzll(exp);
    int bit_len = 63 - leading_zeros;

    uint64_t res = 1;

    // We process bits from LSB to the bit before MSB
    // The loop is structured to allow the compiler to unroll or use 
    // better ILP for the squaring and conditional multiplication.
    for (int i = 0; i < bit_len; ++i) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the leading bit (always 1 at this stage)
    return res * base;
}