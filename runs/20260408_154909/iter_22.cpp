#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases with zero-latency returns where possible
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Fast path for common small exponents to avoid bit-scan overhead
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use __builtin_ctzll to skip initial identity multiplications (res = 1 * base^2^k)
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Initial squaring for the trailing zeros (res = base^(2^trailing_zeros))
    for (int i = 0; i < trailing_zeros; ++i) {
        res *= res;
    }

    uint64_t remaining_exp = exp >> (trailing_zeros + 1);
    if (remaining_exp == 0) return res;

    uint64_t side = res;
    
    // Determine the number of bits remaining to process
    // This allows the compiler to reason about loop bounds better than while(exp)
    int leading_bits = 64 - __builtin_clzll(remaining_exp);

    for (int i = 0; i < leading_bits; ++i) {
        side *= side;
        
        // Use a branchless approach to update res
        // If bit is 0, multiplier is 1. If bit is 1, multiplier is side.
        uint64_t mask = (remaining_exp & 1);
        uint64_t multiplier = (side & -mask) | (1 & (mask - 1));
        res *= multiplier;
        
        remaining_exp >>= 1;
    }

    return res;
}