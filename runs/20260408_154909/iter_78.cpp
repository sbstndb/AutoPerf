#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;

    // Special case for base 2: use shifts.
    // Note: 1ULL << 64 is UB, but standard imul 64-bit wrap-around 
    // for base=2, exp=64 is 0.
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t result = 1;

    // Find the highest set bit to avoid unnecessary iterations
    // __builtin_clzll is undefined for 0, but exp=0 is handled above.
    int leading_zeros = __builtin_clzll(exp);
    int bit_width = 63 - leading_zeros;

    // Binary Exponentiation (Left-to-Right)
    // This approach is often faster on modern CPUs because the 
    // 'result' dependency chain is tighter, and 'base' remains constant.
    // However, Right-to-Left (original) is better for ILP of 'base' squaring.
    // We stick to Right-to-Left but optimize the loop structure.
    
    for (int i = 0; i < bit_width; ++i) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final bit (the MSB is always 1 at this point)
    return result * base;
}