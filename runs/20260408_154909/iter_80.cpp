#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    
    // Use __builtin_ctzll to skip trailing zeros in exponent
    // This reduces the number of squarings before the first '1' bit
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Initial squarings to reach the first set bit
    // We can't easily avoid these as they define the starting base
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Process remaining bits. 
    // We use a fixed-iteration style loop or a simple while to keep ILP high.
    // The compiler will unroll this effectively.
    while (exp > 0) {
        base *= base;
        
        // Branchless update: if bit is 0, multiplier is 1. If bit is 1, multiplier is base.
        // This is often faster than a branch on modern Intel architectures.
        uint64_t mask = (exp & 1);
        uint64_t multiplier = (mask ? base : 1);
        result *= multiplier;
        
        exp >>= 1;
    }

    return result;
}