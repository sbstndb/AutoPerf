#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Find the position of the highest set bit
    // __builtin_clzll returns the number of leading zeros
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;

    // Initialize result with base (representing the MSB)
    uint64_t res = base;

    // Process bits from MSB-1 down to LSB
    // This "top-down" approach is generally faster for integer pow
    // as it minimizes the number of variables to track.
    for (int i = bit_pos - 1; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}