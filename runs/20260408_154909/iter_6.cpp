#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    // Skip trailing zeros in exponent to initialize result
    // and reduce the number of iterations.
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res = 1;
    
    // Square the base for all trailing zero bits
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    res = base;
    exp >>= (trailing_zeros + 1);

    // Standard Binary Exponentiation (Exponentiation by Squaring)
    // for the remaining bits.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}