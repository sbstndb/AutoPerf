#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Skip trailing zeros in exponent to reduce iterations and 
    // initialize 'res' with a higher power of base immediately.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Bottom-up binary exponentiation (Exponentiation by Squaring)
    // This allows the CPU to calculate base *= base in parallel with 
    // the result accumulation due to lower data dependency.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}