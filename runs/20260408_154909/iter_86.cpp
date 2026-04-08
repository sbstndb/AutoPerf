#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Find the highest set bit to determine the number of iterations
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    // Use the trailing zeros to reduce the number of iterations in the main loop
    int trailing_zeros = __builtin_ctzll(exp);
    
    // Start with the base raised to the power of the first bit
    uint64_t res = 1;
    uint64_t side = base;

    // Binary exponentiation (Right-to-left)
    // This version is generally more friendly to ILP than left-to-right
    // because the 'side' squaring sequence is independent of the 'res' updates.
    
    // Process bits from LSB to MSB
    // We iterate up to the highest set bit
    for (int i = 0; i < bit_pos; ++i) {
        if (exp & 1) {
            res *= side;
        }
        side *= side;
        exp >>= 1;
    }
    
    // Final multiplication for the highest bit
    res *= side;

    return res;
}