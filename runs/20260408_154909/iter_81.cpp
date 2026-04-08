#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Use bit scan to find the highest set bit.
    // This tells us exactly how many iterations we need, 
    // eliminating the 'while (exp > 1)' branch uncertainty.
    int iterations = 63 - __builtin_clzll(exp);
    
    uint64_t result = (exp & 1) ? base : 1;
    
    // We process bits from LSB to MSB.
    // The loop is structured to allow the compiler to unroll or 
    // pipeline the squaring and the result multiplication.
    for (int i = 0; i < iterations; ++i) {
        exp >>= 1;
        base *= base;
        
        // Use a data-dependency instead of a branch to update result.
        // This typically compiles to a TEST + CMOV or a simple mask.
        uint64_t multiplier[2] = {1, base};
        result *= multiplier[exp & 1];
    }

    return result;
}