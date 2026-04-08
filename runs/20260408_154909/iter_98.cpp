#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp 0 is 1, base 0 is 0, base 1 is 1
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2 (Shift is significantly faster)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Bottom-up Binary Exponentiation (Exponentiation by Squaring)
    // This method is generally faster as it allows for better ILP 
    // between the 'res' update and the 'base' squaring.
    
    // Skip trailing zeros to reduce iterations and initialize res
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Process remaining bits
    while (exp > 0) {
        base *= base;
        
        // Use a branchless approach to update res. 
        // The compiler typically optimizes this to a CMOV or 
        // a sequence of instructions without a branch.
        uint64_t next_res = res * base;
        if (exp & 1) {
            res = next_res;
        }
        
        exp >>= 1;
    }

    return res;
}