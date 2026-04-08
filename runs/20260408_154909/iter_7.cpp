#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Use __builtin_ctzll to find the first set bit and skip initial identity multiplications
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Initial squaring for the trailing zeros
    for (int i = 0; i < trailing_zeros; ++i) {
        res *= res;
    }

    exp >>= (trailing_zeros + 1);
    uint64_t side = res;

    // Binary exponentiation loop
    while (exp > 0) {
        side *= side; // Square the base
        if (exp & 1) {
            res *= side; // Multiply result if bit is set
        }
        exp >>= 1;
    }

    return res;
}