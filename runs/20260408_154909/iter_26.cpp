#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use trailing zeros to jump-start the 'res' value
    // This reduces the number of iterations in the main loop
    int tz = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Square the base for every trailing zero in the exponent
    for (int i = 0; i < tz; ++i) {
        res *= res;
    }

    uint64_t remaining_exp = exp >> (tz + 1);
    if (remaining_exp == 0) return res;

    uint64_t side = res;
    uint64_t final_res = res;

    // Main Binary Exponentiation Loop
    // We use a fixed number of iterations based on the leading bit 
    // to encourage the compiler to unroll or use better scheduling.
    int leading_bits = 63 - __builtin_clzll(remaining_exp);
    
    for (int i = 0; i <= leading_bits; ++i) {
        side *= side;
        if (remaining_exp & 1) {
            final_res *= side;
        }
        remaining_exp >>= 1;
    }

    return final_res;
}