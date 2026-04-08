#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Skip trailing zeros in exponent to reduce iterations and 
    // initialize 'res' with the first required power of base.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Pre-scale base to the first set bit
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Bottom-up binary exponentiation (Square-and-Multiply)
    // This allows the CPU to calculate base^2 and res*base in parallel.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}