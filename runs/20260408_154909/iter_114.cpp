#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // Standard behavior for 64-bit unsigned: 2^64 is 0 (wrap around)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Skip trailing zeros in exponent to initialize 'res'
    // This reduces the number of iterations and avoids res = 1 initialization
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Pre-square base for the skipped bits using a small unrolled sequence
    // This is faster than a loop for the typical small number of trailing zeros
    while (trailing_zeros >= 2) {
        base *= base;
        base *= base;
        trailing_zeros -= 2;
    }
    if (trailing_zeros) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Bottom-up binary exponentiation (Square-and-Multiply)
    // We use a branchless approach to keep the pipeline full.
    // The compiler will typically map the ternary to a CMOV instruction.
    while (exp > 0) {
        base *= base;
        
        uint64_t multiplier = (exp & 1) ? base : 1;
        res *= multiplier;
        
        exp >>= 1;
        
        // Manual unroll for the next bit to increase ILP
        if (exp == 0) break;
        
        base *= base;
        multiplier = (exp & 1) ? base : 1;
        res *= multiplier;
        exp >>= 1;
    }

    return res;
}