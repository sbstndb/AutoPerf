#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common edge cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    
    // Skip trailing zeros in exponent to reduce iterations
    // and initialize result with the first required base power.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: process remaining bits
    while (exp > 0) {
        base *= base;
        
        // Use a temporary to help the compiler avoid branching.
        // The multiplication happens every time, but the result is 
        // only updated if the bit is set.
        uint64_t side_effect = result * base;
        if (exp & 1) result = side_effect;
        
        exp >>= 1;
        
        // Manual unroll for the next bit to improve ILP
        if (exp == 0) break;
        
        base *= base;
        side_effect = result * base;
        if (exp & 1) result = side_effect;
        
        exp >>= 1;
    }

    return result;
}