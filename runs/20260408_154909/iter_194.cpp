#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case for base 2: use bit shift (extremely fast)
    if (base == 2) {
        return (exp >= 64) ? 0 : (1ULL << exp);
    }

    // Count trailing zeros to reduce the number of iterations
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero. 
    // Unrolling manually or letting compiler handle small loop.
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Main Binary Exponentiation Loop
    // We use a technique to minimize branch misprediction penalties.
    while (exp > 0) {
        base *= base;
        
        // Use a conditional to only multiply when bit is set.
        // On modern x86, the compiler often optimizes this to a CMOV 
        // or a sequence that avoids a hard-to-predict branch.
        if (exp & 1) {
            res *= base;
        }
        
        exp >>= 1;
    }

    return res;
}