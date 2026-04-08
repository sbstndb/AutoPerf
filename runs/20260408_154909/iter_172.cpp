#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case for base 2: use bit shift
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    // Find the first set bit to initialize 'res' and 'b'
    // This replaces the trailing zero loop and the initial 'res = 1' logic
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t b = base;
    uint64_t res = 1;

    // Standard Binary Exponentiation (Right-to-Left)
    // This version is often more ILP-friendly than Left-to-Right
    // because the squaring of 'b' can happen independently of 'res' updates.
    while (true) {
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
        if (exp == 0) break;
        b *= b;

        // Unroll once to reduce branch overhead and increase ILP
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
        if (exp == 0) break;
        b *= b;
    }

    return res;
}