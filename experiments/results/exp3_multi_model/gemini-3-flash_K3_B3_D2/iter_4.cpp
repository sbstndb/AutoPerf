#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: handle common small bases and exponents immediately
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        return base; // base is 0 or 1
    }

    uint64_t result = 1;

    // Remove trailing zeros from exponent to reduce iterations and 
    // initialize 'result' with the first required power of 'base'.
    // This avoids one 'result *= 1' operation.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    while (trailing_zeros--) {
        base *= base;
    }
    result = base;
    exp >>= 1;

    // Main Exponentiation by Squaring loop
    // Unrolled slightly to improve Instruction Level Parallelism (ILP)
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        
        if (exp == 0) break;

        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}