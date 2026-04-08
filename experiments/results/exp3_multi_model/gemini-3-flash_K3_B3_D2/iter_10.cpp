#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    // Skip trailing zeros to reduce iterations and initialize result
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for the initial trailing zeros
    // This is often faster than starting result at 1
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Process remaining bits
    // We use a branchless approach to keep the pipeline full
    while (exp > 0) {
        base *= base;
        
        // Use a conditional multiplier to avoid branching
        // If (exp & 1) is 0, multiplier is 1. If 1, multiplier is base.
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;
        
        exp >>= 1;
        
        // Optimization: if exp is small, we can finish early
        // but the overhead of checking often outweighs the benefit.
        // The compiler will likely generate a CMOV for the multiplier.
    }

    return result;
}