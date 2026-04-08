#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for identity and zero cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    uint64_t result = 1;

    // Determine the highest bit set to avoid unnecessary iterations
    // __builtin_clzll is a hardware-accelerated instruction (BSR/LZCNT)
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    // Unroll the loop slightly to allow the CPU to pipeline the imul instructions.
    // We process the bits from LSB to MSB.
    for (int i = 0; i < iterations; ++i) {
        // Branchless update: result is multiplied by base only if (exp & 1)
        // This helps the execution pipeline stay full.
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last bit (which is always 1 after the loop)
    return result * base;
}