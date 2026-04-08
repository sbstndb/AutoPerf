#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for very small exponents - covers most common cases immediately
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for small bases to prevent unnecessary loops
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Use Bit Scan Reverse to find the highest set bit.
    // This allows us to skip all leading zeros in the exponent.
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t res = 1;

    // We process the bits from most-significant to least-significant.
    // This is the "Left-to-Right" binary exponentiation algorithm.
    // It is often faster on modern CPUs because the 'res' dependency chain
    // is more consistent.
    
    // Start with the bit below the highest set bit
    for (int i = bit_pos; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}