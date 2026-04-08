#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases efficiently
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Skip trailing zeros in exponent: base^(binary ...1000) = (base^8)^...
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res_base = base;
    
    // Square the base for each trailing zero
    for (int i = 0; i < trailing_zeros; ++i) {
        res_base *= res_base;
    }

    uint64_t res = res_base;
    exp >>= (trailing_zeros + 1);

    // If no bits left, we are done
    if (exp == 0) return res;

    // Binary exponentiation loop
    // We use a multiplier array to eliminate branching for the 'if (exp & 1)' check
    // This is often faster than a branch on modern out-of-order CPUs
    uint64_t multiplier[2] = {1, 0}; 

    while (exp > 0) {
        res_base *= res_base;
        
        // Use a conditional multiplier to avoid branching
        // The compiler typically optimizes this to a CMOV (Conditional Move)
        uint64_t m = (exp & 1) ? res_base : 1;
        res *= m;
        
        exp >>= 1;
        if (exp == 0) break;

        // Unroll once more to increase ILP
        res_base *= res_base;
        m = (exp & 1) ? res_base : 1;
        res *= m;
        exp >>= 1;
    }

    return res;
}