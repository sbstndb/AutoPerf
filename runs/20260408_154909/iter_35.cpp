#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Early exits for common or edge cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Use __builtin_ctzll to find the first set bit and initialize res.
    // This avoids multiplying res by 1 in the first few iterations.
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Square the base to match the first set bit of the exponent.
    // This is faster than the loop in the original code for small exponents.
    for (int i = 0; i < trailing_zeros; ++i) {
        res *= res;
    }
    
    uint64_t current_base = res;
    exp >>= (trailing_zeros + 1);

    // If no bits are left in the exponent, we are done.
    if (exp == 0) return res;

    // Standard Binary Exponentiation (Exponentiation by Squaring).
    // We use a fixed number of iterations based on the remaining bits
    // to allow the CPU to better predict the loop exit.
    int remaining_bits = 64 - __builtin_clzll(exp);
    for (int i = 0; i < remaining_bits; ++i) {
        current_base *= current_base;
        if (exp & 1) {
            res *= current_base;
        }
        exp >>= 1;
    }

    return res;
}