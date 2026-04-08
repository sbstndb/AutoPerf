#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base 0 case immediately
    if (base == 0) return exp == 0 ? 1 : 0;
    // Handle exp 0 case immediately
    if (exp == 0) return 1;

    // Skip trailing zeros in exponent to initialize result
    // This avoids result = 1 * base and goes straight to result = base
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Branchless exponentiation by squaring
    while (exp > 0) {
        base *= base;
        // Use a temporary to avoid branching on (exp & 1)
        // The compiler will typically optimize this to a test and cmov
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;
        exp >>= 1;
    }

    return result;
}