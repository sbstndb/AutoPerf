#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases: 0^exp, 1^exp, base^0
    if (exp == 0) return 1;
    if (base <= 1) return base;

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation (Right-to-Left)
    // We use a pattern that encourages the compiler to use CMOV 
    // to avoid branch mispredictions on the bits of the exponent.
    while (exp > 1) {
        // Bit 0
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;

        // Bit 1 (Unrolled to reduce branch frequency and increase ILP)
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;
        
        // The CPU can overlap the squaring of 'b' with the 
        // conditional multiplication of 'res'.
    }

    // Final bit
    if (exp) res *= b;

    return res;
}