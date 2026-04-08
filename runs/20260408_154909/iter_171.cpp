#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^0 = 1, x^0 = 1
    if (exp == 0) return 1;
    // 0^x = 0, 1^x = 1
    if (base <= 1) return base;
    
    // Special case for base 2: use bit shift (SHLX/SHL)
    // Handles 64-bit overflow naturally (returns 0 for exp >= 64)
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation (Square and Multiply)
    // We use a pattern that encourages the compiler to use CMOV (Conditional Move)
    // to avoid branch misprediction penalties which are costly in tight loops.
    while (true) {
        // If bit is set, multiply result by current base power
        if (exp & 1) res *= b;
        
        exp >>= 1;
        if (exp == 0) break;
        
        // Square the base for the next bit
        b *= b;

        // Unroll once to reduce loop overhead and improve ILP
        if (exp & 1) res *= b;
        
        exp >>= 1;
        if (exp == 0) break;
        
        b *= b;
    }

    return res;
}