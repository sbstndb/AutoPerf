#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: handle early to avoid overhead in the hot path
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Use the "Bottom-Up" Binary Exponentiation (Exponentiation by Squaring)
    // This approach is more efficient for ILP as base squaring and 
    // result accumulation can be pipelined.
    uint64_t res = 1;

    // Eliminate trailing zeros to reduce the number of iterations
    // and potentially skip the loop entirely for powers of 2.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;

    // Square the base for every trailing zero bit
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    // Now exp is guaranteed to be odd, so we can handle the first bit 
    // outside the loop to initialize 'res' and avoid one multiplication.
    res = base;
    exp >>= 1;

    // Main loop: Process remaining bits
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}