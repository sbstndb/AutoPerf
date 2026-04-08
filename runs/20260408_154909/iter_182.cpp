#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases: 0^exp, 1^exp, base^0
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Count trailing zeros to skip initial multiplications where res would be 1
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Initial squaring for trailing zeros
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Determine the number of bits remaining to process
    // This allows us to use a fixed-count loop or avoid extra checks
    int remaining_bits = 64 - __builtin_clzll(exp);

    // Main loop: Square and Multiply (Binary Exponentiation)
    // Using a branchless approach to multiply res by b only if the bit is set
    for (int i = 0; i < remaining_bits; ++i) {
        b *= b;
        uint64_t mask = -(exp & 1);
        // If bit is 1, mask is all 1s (0xFF...FF), res *= b
        // If bit is 0, mask is 0, res *= 1
        res *= (b & mask) | (1ULL & ~mask);
        exp >>= 1;
    }

    return res;
}