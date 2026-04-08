#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents (0, 1, 2)
    // Using a single comparison to cover the range
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2
    // Base 2 is optimized using SHLX (Shift Logical Left) via the ternary
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Use Bit Scan Reverse to find the highest set bit.
    // This tells us exactly how many squarings are needed.
    uint32_t leading_zeros = __builtin_clzll(exp);
    uint32_t bits = 63 - leading_zeros;
    
    uint64_t res = 1;

    // Unroll the loop slightly or use a structure that allows the CPU 
    // to pipeline the 'base * base' independent of the 'res * base'.
    // We process from LSB to the bit before the MSB.
    for (uint32_t i = 0; i < bits; ++i) {
        // Branchless update: if bit is set, multiply res by current base.
        // Modern CPUs execute this pattern very efficiently.
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The final multiplication by the 'base' representing the MSB.
    // Since the MSB is always 1 for exp > 0, we don't need a branch.
    return res * base;
}