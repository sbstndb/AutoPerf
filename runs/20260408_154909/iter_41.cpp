#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use __builtin_clzll to find the highest set bit.
    // This allows us to avoid unnecessary iterations and squaring.
    // 63 - clz gives the index of the highest bit.
    int iterations = 63 - __builtin_clzll(exp);

    // Unroll the loop manually to reduce branch overhead and 
    // allow the compiler to use CMOV (Conditional Move) instructions.
    // This prevents branch mispredictions for the (exp & 1) check.
    for (int i = 0; i < iterations; ++i) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The final multiplication for the highest bit
    // (which is always 1 at this point)
    return result * base;
}