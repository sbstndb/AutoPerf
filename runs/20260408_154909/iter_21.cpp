#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common/simplest cases first with minimal branching
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    uint64_t result = 1;

    // Optimization: Skip trailing zeros in the exponent to reduce iterations.
    // This handles the 'squaring' phase before the first 'multiplication' phase.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for all trailing zeros
    while (trailing_zeros--) {
        base *= base;
    }

    // Binary Exponentiation with reduced branching
    // We check exp > 1 because the final bit is handled outside to save one imul
    while (exp > 1) {
        // Use a mask to avoid a hard branch. 
        // If (exp & 1) is 0, temp becomes 1 (no-op for imul).
        // If (exp & 1) is 1, temp becomes base.
        uint64_t mask = -static_cast<int64_t>(exp & 1);
        uint64_t temp = (base & mask) | (1ULL & ~mask);
        result *= temp;

        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last remaining bit (always 1 at this point)
    return result * base;
}