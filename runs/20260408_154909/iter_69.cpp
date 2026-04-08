#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros in exponent by pre-squaring the base.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // After shifting out trailing zeros, the LSB of exp is guaranteed to be 1.
    // Initialize result with base and move to the next bit.
    uint64_t result = base;
    exp >>= 1;

    // Standard Binary Exponentiation (Square and Multiply).
    // Simple branching is often faster than complex bit-manipulation 
    // on modern x86 due to excellent branch prediction and lower ILP pressure.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}