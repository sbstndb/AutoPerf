#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: handle the most frequent/simplest first
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        return 0; // base is 0
    }

    uint64_t result = 1;

    // Skip trailing zeros in exponent to reduce iterations
    // and initialize 'result' with the first power of base.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    while (trailing_zeros--) {
        base *= base;
    }
    result = base;
    exp >>= 1;

    // Main loop: Exponentiation by squaring with manual unrolling
    // We use the fact that base *= base and result *= base can be pipelined.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        
        // Optional: Unroll once more if exp is still large
        if (exp == 0) break;
        
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}