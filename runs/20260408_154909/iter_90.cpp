#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Special case for base 2: use bit shift (very fast)
    // Note: result for 2^64+ will overflow to 0, matching multiplication behavior
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    // Use trailing zeros to skip initial squaring/multiplication steps
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    
    // Square the base for each trailing zero to get the first 'res'
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }

    uint64_t res = side;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Determine the number of bits remaining to process
    // This allows the compiler to unroll the loop better than 'while(exp)'
    int remaining_bits = 63 - __builtin_clzll(exp);

    for (int i = 0; i <= remaining_bits; ++i) {
        side *= side;
        
        // Use a mask to perform a conditional multiplication without a branch.
        // If (exp & 1) is 0, multiplier becomes 1. If 1, multiplier becomes side.
        // This improves ILP by removing branch misprediction penalties.
        uint64_t mask = -(uint64_t)(exp & 1);
        uint64_t multiplier = (side & mask) | (1ULL & ~mask);
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}