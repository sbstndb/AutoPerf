#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^exp, 1^exp, base^0
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2 (2^64 wraps to 0 in 64-bit unsigned)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Find the position of the highest set bit to determine loop bounds
    // This avoids processing leading zeros of the exponent
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t res = 1;
    
    // Unroll the loop to improve Instruction Level Parallelism (ILP)
    // We process bits from LSB to MSB for bottom-up binary exponentiation
    // but we only go up to the highest set bit.
    while (exp > 0) {
        // Branchless update: res *= (exp & 1) ? base : 1;
        // The compiler optimizes this to a test + cmov or a mask-based imul
        uint64_t mask = -(exp & 1);
        uint64_t multiplier = (base & mask) | (1ULL & ~mask);
        res *= multiplier;

        // Square the base for the next bit
        base *= base;
        exp >>= 1;

        // Manual unrolling for the next bit to allow squaring and 
        // result multiplication to happen in parallel pipelines
        if (exp == 0) break;

        mask = -(exp & 1);
        multiplier = (base & mask) | (1ULL & ~mask);
        res *= multiplier;

        base *= base;
        exp >>= 1;
    }

    return res;
}