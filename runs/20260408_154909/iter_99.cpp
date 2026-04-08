#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle identity cases early
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Skip trailing zeros in the exponent to reduce iterations.
    // This effectively transforms base^exp into (base^(2^ctz))^ (exp >> ctz)
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for every trailing zero
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = 1;

    // Right-to-Left binary exponentiation
    // This allows the CPU to calculate base *= base and result *= base
    // with more overlapping execution (ILP) than the Left-to-Right method.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last remaining bit of exp (which is always 1 here)
    return result * base;
}