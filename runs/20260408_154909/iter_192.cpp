#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^0 is 1, 1^n is 1, 0^n is 0
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Count trailing zeros to reduce the number of iterations in the main loop
    // and start with a larger base.
    int tz = __builtin_ctzll(exp);
    uint64_t e = exp >> tz;
    
    // Square the base for all trailing zeros.
    // Unroll slightly to improve ILP during the squaring phase.
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz & 1) {
        base *= base;
    }

    uint64_t res = base;
    e >>= 1;

    // Main Exponentiation by Squaring loop.
    // We use a branchless approach for the result accumulation to avoid
    // pipeline stalls from unpredictable exponent bits.
    while (e > 0) {
        base *= base;
        
        // Branchless update: if bit is 0, multiplier is 1; if bit is 1, multiplier is base.
        // This is often faster than a branch for random exponents.
        uint64_t side = (e & 1);
        uint64_t multiplier = (base & -side) | (1 & (side - 1));
        res *= multiplier;
        
        e >>= 1;
    }

    return res;
}