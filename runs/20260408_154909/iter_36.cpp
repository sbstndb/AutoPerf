#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    uint64_t res = 1;

    // Use __builtin_ctzll to skip trailing zeros in the exponent
    // This reduces the number of iterations and initializes 'res' 
    // with a power of base instead of 1.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Fast-forward base to the first set bit
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    res = base;
    exp >>= 1;

    // Process remaining bits
    while (exp > 0) {
        base *= base;
        
        // Use a conditional multiply to avoid branching.
        // The compiler typically optimizes this to a test + cmov + imul
        // or a sequence that avoids the pipeline flush of a mispredicted branch.
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}