#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use tzcnt to skip trailing zeros in the exponent (powers of 2)
    // This reduces the number of iterations and initializes 'res' effectively.
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Square the base for each trailing zero bit
    for (int i = 0; i < trailing_zeros; ++i) {
        res *= res;
    }

    // Shift out the trailing zeros and the first '1' bit
    uint64_t remaining_exp = exp >> (trailing_zeros + 1);
    if (remaining_exp == 0) return res;

    uint64_t side = res;
    
    // Standard Binary Exponentiation (Square-and-Multiply)
    // We use a loop that executes exactly as many times as there are bits left.
    int leading_bits = 63 - __builtin_clzll(remaining_exp);
    for (int i = 0; i <= leading_bits; ++i) {
        side *= side;
        if (remaining_exp & 1) {
            res *= side;
        }
        remaining_exp >>= 1;
    }

    return res;
}