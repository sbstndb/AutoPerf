#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp 0 is always 1, base 0/1 are fixed points.
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Handle trailing zeros to jumpstart the 'res' value.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero.
    // Unrolling this manually or using a small loop is efficient.
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Main Exponentiation by Squaring loop.
    // We unroll the loop to improve ILP and use a branchless 
    // update for 'res' to avoid pipeline stalls.
    while (exp >= 2) {
        // Bit 0
        base *= base;
        uint64_t mask0 = -(exp & 1);
        res *= (base & mask0) | (1ULL & ~mask0);
        
        // Bit 1
        base *= base;
        uint64_t mask1 = -((exp >> 1) & 1);
        res *= (base & mask1) | (1ULL & ~mask1);

        exp >>= 2;
    }

    // Final bit if exp was odd
    if (exp) {
        base *= base;
        res *= base;
    }

    return res;
}