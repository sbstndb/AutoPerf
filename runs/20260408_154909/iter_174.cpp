#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    
    // Special case for base 2: use bit shift
    // Handles overflow naturally via 64-bit shift rules
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation (Square and Multiply)
    // We use a branchless approach to update 'res' to avoid mispredictions.
    // The loop is unrolled to improve ILP.
    while (true) {
        // Bit 0
        uint64_t mask0 = -(exp & 1);
        res *= (b & mask0) | (1ULL & ~mask0);
        b *= b;
        exp >>= 1;
        if (exp == 0) break;

        // Bit 1 (Unrolled)
        uint64_t mask1 = -(exp & 1);
        res *= (b & mask1) | (1ULL & ~mask1);
        b *= b;
        exp >>= 1;
        if (exp == 0) break;
        
        // Bit 2 (Unrolled)
        uint64_t mask2 = -(exp & 1);
        res *= (b & mask2) | (1ULL & ~mask2);
        b *= b;
        exp >>= 1;
        if (exp == 0) break;

        // Bit 3 (Unrolled)
        uint64_t mask3 = -(exp & 1);
        res *= (b & mask3) | (1ULL & ~mask3);
        b *= b;
        exp >>= 1;
        if (exp == 0) break;
    }

    return res;
}