#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2 to handle potential overflows or speed up
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    uint64_t res = 1;
    
    // Standard Binary Exponentiation (Square and Multiply)
    // Modern x86 CPUs (Golden Cove/Redwood Cove) have excellent branch prediction.
    // The "branchless" mask approach often adds more latency than a predicted branch.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the leading bit
    return res * base;
}