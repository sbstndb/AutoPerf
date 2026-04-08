#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common cases
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 0) return 0;
        if (base == 1) return 1;
        // base == 2
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // Skip trailing zeros in exponent to reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for the initial trailing zeros
    // This is faster than starting res at 1 and looping
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    
    // Binary Exponentiation (LSB to MSB)
    // This approach allows for better Instruction Level Parallelism (ILP)
    // as 'base' and 'res' updates can be pipelined.
    while (exp > 0) {
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break; // Avoid unnecessary squaring on last bit
        base *= base;
        
        // Manual unroll for next bit to improve ILP
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        base *= base;
    }

    return res;
}