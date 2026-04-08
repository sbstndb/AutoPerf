#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp 0 is always 1, base 0/1 are fixed.
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Handle base 2 as a special case for extreme performance
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    // Remove trailing zeros to jump-start the 'res' value.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square the base for each trailing zero.
    // Unrolling this slightly helps with ILP.
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    uint64_t res = base;
    exp >>= 1;

    // Main Exponentiation by Squaring loop.
    // We use a branchless-style update to keep the pipeline full.
    // On modern Intel cores, imul has a latency of 3 cycles but high throughput.
    while (exp > 0) {
        base *= base;
        
        // Use a conditional move or simple multiplier selection to avoid 
        // branch mispredictions on random exponent bits.
        uint64_t side = (exp & 1) ? base : 1;
        res *= side;
        
        exp >>= 1;
    }

    return res;
}