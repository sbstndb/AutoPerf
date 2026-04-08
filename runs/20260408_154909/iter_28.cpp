#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common/trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    
    // Special case for base 2: use bit shift (much faster)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // Binary Exponentiation (Exponentiation by Squaring)
    // We iterate only up to the highest set bit of exp
    int bits = 64 - __builtin_clzll(exp);
    
    // Unroll the first few iterations or use a branchless approach
    // to minimize the cost of the loop.
    for (int i = 0; i < bits; ++i) {
        // Branchless update: 
        // If the i-th bit is set, multiply res by base.
        // Otherwise, multiply res by 1 (no-op).
        uint64_t mask = -static_cast<uint64_t>((exp >> i) & 1);
        uint64_t multiplier = (base & mask) | (1ULL & ~mask);
        res *= multiplier;
        
        // Square the base for the next bit
        // We avoid squaring on the very last iteration to save one IMUL
        if (i < bits - 1) {
            base *= base;
        }
    }

    return res;
}