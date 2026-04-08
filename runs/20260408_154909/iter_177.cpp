#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Find the highest set bit to avoid unnecessary iterations.
    // __builtin_clzll is undefined for 0, but exp == 0 is handled above.
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;

    uint64_t res = 1;
    uint64_t b = base;

    // Standard Square-and-Multiply (Binary Exponentiation)
    // We iterate from LSB to MSB as it allows the squaring (b *= b)
    // to happen independently of the result multiplication in some cycles.
    while (exp > 0) {
        // Use a conditional to multiply. On modern x86, the compiler 
        // will often optimize this to avoid a branch if it's profitable.
        if (exp & 1) {
            res *= b;
        }
        
        // Prepare for next bit
        exp >>= 1;
        if (exp == 0) break; // Optimization: avoid the last squaring
        
        b *= b;

        // Manual unroll: Process second bit to reduce loop overhead
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
        if (exp == 0) break;
        
        b *= b;
    }

    return res;
}