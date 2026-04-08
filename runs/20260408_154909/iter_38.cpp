#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    uint64_t result = 1;

    // Use __builtin_ctzll to skip trailing zeros in the exponent
    // This reduces the number of iterations and initializes 'result' efficiently
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Instead of starting with result=1 and squaring base many times,
    // we can jumpstart the base to the first set bit.
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    
    // Process the remaining bits. 
    // We use a branchless approach to update the result.
    while (exp > 0) {
        // If bit is set, multiply result by current base.
        // If bit is not set, multiply result by 1 (no-op for value, but keeps pipeline full).
        // However, a simple if is often converted to a CMOV by modern GCC.
        if (exp & 1) {
            result *= base;
        }
        
        exp >>= 1;
        if (exp == 0) break; // Optimization: don't square base on the very last bit
        base *= base;
    }

    return result;
}