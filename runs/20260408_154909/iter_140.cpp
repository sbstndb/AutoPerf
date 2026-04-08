#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents (0, 1, 2)
    // This covers a large percentage of calls in typical workloads.
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2
    // Base 2 is optimized to a single shift instruction.
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Use bit scanning to find the highest bit set.
    // This allows us to skip leading zeros and reduces loop overhead.
    int leading_zeros = __builtin_clzll(exp);
    int bit_width = 63 - leading_zeros;
    
    uint64_t res = 1;
    
    // We process the exponent from the most significant bit to the least.
    // This is the "Left-to-Right" binary exponentiation algorithm.
    // It is often faster on modern CPUs because 'res' is updated more predictably.
    for (int i = bit_width; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}