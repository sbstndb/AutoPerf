#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases: 0^exp, 1^exp, base^0
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Use trailing zero count to jump-start the base squaring.
    // This effectively handles the first '1' bit in the exponent.
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Unrolled squaring for trailing zeros
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    // Main loop: Square and Multiply (Binary Exponentiation)
    // We unroll manually to allow the CPU to overlap squaring and bit-testing.
    while (exp > 0) {
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
        
        // Secondary unroll to reduce branch frequency
        if (exp == 0) break;
        
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
    }

    return res;
}