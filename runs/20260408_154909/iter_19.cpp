#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Skip trailing zeros in the exponent to reduce iterations
    // This handles cases like exp=16, 32, etc., very quickly.
    int trailing_zeros = __builtin_ctzll(exp);
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    exp >>= trailing_zeros;

    // Binary Exponentiation with unrolling and branchless logic
    while (exp > 1) {
        // If bit is set, multiply result by base
        if (exp & 1) {
            result *= base;
        }
        
        // Square the base
        base *= base;
        exp >>= 1;

        // Manual unroll: check next bit
        if (exp <= 1) break;
        
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    return result * base;
}