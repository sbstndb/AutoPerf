#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;

    // Skip trailing zeros to reduce iterations and initialize result
    // This effectively performs: result = base^(2^first_set_bit)
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for the initial trailing zeros
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Exponentiation by squaring
    // We use a small lookup array to eliminate the branch inside the loop.
    // The compiler will likely optimize this into a CMOV or simple arithmetic.
    while (exp > 0) {
        base *= base;
        uint64_t side_effect[2] = {1, base};
        result *= side_effect[exp & 1];
        exp >>= 1;
    }

    return result;
}