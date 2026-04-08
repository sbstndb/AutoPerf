#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle identity cases immediately
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    
    // Special case: base 2 is a simple shift
    // This is significantly faster and common in many workloads
    if (base == 2) {
        return (exp >= 64) ? 0 : (1ULL << exp);
    }

    // Use Top-Down Binary Exponentiation
    // This avoids the need to handle trailing zeros separately
    // and allows for a cleaner loop structure.
    
    // Find the position of the highest set bit
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t res = base;
    
    // Process bits from second-highest to lowest
    for (int i = bit_pos - 1; i >= 0; --i) {
        res *= res;
        // Branchless update: multiply by base if bit i is set
        // Modern compilers optimize this into a CMOV or a simple jump
        // which is well-predicted for most exponent patterns.
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}