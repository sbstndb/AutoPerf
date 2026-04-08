#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Handle trailing zeros to reduce the exponent magnitude
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }
    exp >>= tz;
    if (exp == 1) return side;

    // 4-bit window (k-ary) exponentiation
    // Precompute powers: side^1, side^2, ..., side^15
    uint64_t table[16];
    table[0] = 1;
    table[1] = side;
    
    uint64_t s2 = side * side;
    uint64_t s4 = s2 * s2;
    uint64_t s8 = s4 * s4;

    table[2] = s2;
    table[3] = s2 * side;
    table[4] = s4;
    table[5] = s4 * side;
    table[6] = s4 * s2;
    table[7] = s4 * s2 * side;
    table[8] = s8;
    table[9] = s8 * side;
    table[10] = s8 * s2;
    table[11] = s8 * s2 * side;
    table[12] = s8 * s4;
    table[13] = s8 * s4 * side;
    table[14] = s8 * s4 * s2;
    table[15] = s8 * s4 * s2 * side;

    int high_bit = 63 - __builtin_clzll(exp);
    
    // Start with the highest window
    int shift = (high_bit / 4) * 4;
    uint64_t res = table[exp >> shift];
    shift -= 4;

    // Process remaining windows
    while (shift >= 0) {
        // 4 squarings (unrolled)
        res *= res;
        res *= res;
        res *= res;
        res *= res;
        
        // Multiply by the precomputed power for this window
        res *= table[(exp >> shift) & 0xF];
        shift -= 4;
    }

    // Handle remaining bits if high_bit was not a multiple of 4
    // This is implicitly handled by the shift logic if we adjust the start,
    // but for 64-bit integers, a simple cleanup or bit-by-bit for the last <4 bits works.
    // However, the logic above covers all bits if we adjust the initial 'res'.
    
    return res;
}