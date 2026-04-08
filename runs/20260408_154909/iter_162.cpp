#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle trailing zeros in exponent: base^exp = (base^(2^tz))^(exp >> tz)
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }
    exp >>= tz;

    if (exp == 1) return base;

    // Standard Binary Exponentiation (Square-and-Multiply)
    // We start from the highest set bit.
    uint64_t res = base;
    int bit_pos = 62 - __builtin_clzll(exp);

    // Process bits from most significant to least significant
    for (int i = bit_pos; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}