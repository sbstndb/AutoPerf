#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;
    
    // Special case for base 2 (common in bit manipulation)
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    uint64_t res = 1;
    uint64_t side = base;

    // Determine the number of bits to process
    int bit_pos = 63 - __builtin_clzll(exp);

    // Right-to-left binary exponentiation with unrolling
    // We process bits in chunks to improve ILP.
    // The 'side' squaring chain is the bottleneck; unrolling helps the scheduler.
    while (bit_pos >= 2) {
        // Bit 0
        if (exp & 1) res *= side;
        side *= side;
        
        // Bit 1
        if (exp & 2) res *= side;
        side *= side;

        exp >>= 2;
        bit_pos -= 2;
    }

    // Handle remaining bits
    if (bit_pos > 0) {
        if (exp & 1) res *= side;
        side *= side;
        exp >>= 1;
    }

    // The final bit (MSB) is always 1 because of the bit_pos calculation
    // This final multiplication completes the result.
    res *= side;

    return res;
}