#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Standard identity: x^0 = 1
    if (exp == 0) return 1;
    // Fast path for exp == 1
    if (exp == 1) return base;

    uint64_t result = 1;

    // Find the position of the highest set bit to determine the number of iterations.
    // __builtin_clzll is a hardware-accelerated instruction (BSR/LZCNT).
    // Since exp > 0, clzll is well-defined.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    // Binary exponentiation (Square and Multiply)
    // We process bits from LSB to MSB-1.
    // The loop is structured to allow the compiler to unroll and use 
    // conditional moves or simple multiplications to avoid branches.
    for (int i = 0; i < iterations; ++i) {
        uint64_t bit = exp & 1;
        // If bit is 1, result *= base; if 0, result *= 1.
        // This is often faster than a branch on modern superscalar CPUs.
        if (bit) result *= base;
        base *= base;
        exp >>= 1;
    }

    // The final bit (the most significant bit) is always 1.
    return result * base;
}