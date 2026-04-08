#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for exponent 0
    if (exp == 0) return 1;
    
    // Optimization: Skip leading zeros to reduce loop iterations.
    // __builtin_clzll is a compiler intrinsic for the BSR/LZCNT instruction.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    uint64_t result = 1;

    // We process the bits from LSB to MSB.
    // To maximize throughput, we use a branchless approach for the result update.
    for (int i = 0; i < iterations; ++i) {
        // Branchless update: if bit is 0, multiply by 1; if 1, multiply by base.
        // This is often optimized by the compiler into a TEST and CMOV.
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The final bit (the highest set bit) always results in result *= base.
    // We do this outside the loop to save one 'base *= base' multiplication.
    return result * base;
}