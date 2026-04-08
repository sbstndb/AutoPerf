#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    
    // Special case for base 2: use bit shift
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Standard Binary Exponentiation (Square and Multiply)
    // We use a branchless approach for updating 'res' to avoid mispredictions.
    // The loop is unrolled to improve ILP.
    while (exp > 1) {
        // Bit 0
        uint64_t next_res = res * b;
        if (exp & 1) res = next_res;
        b *= b;
        exp >>= 1;

        // Bit 1 (Unrolled)
        if (exp <= 1) break;
        next_res = res * b;
        if (exp & 1) res = next_res;
        b *= b;
        exp >>= 1;
    }

    // Final multiply for the last remaining bit
    return res * b;
}