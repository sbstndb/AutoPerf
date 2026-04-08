#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle identity cases
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base == 2) return 1ULL << exp; // Optimization for power of 2
    if (base <= 1) return base;        // 0^n = 0, 1^n = 1

    // Use __builtin_clzll to find the leading bit.
    // This allows us to process the exponent from the second-highest bit downwards.
    // This is the "Square-and-Multiply" algorithm.
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 62 - leading_zeros;
    
    uint64_t res = base;

    // Process bits from most significant to least significant
    // This structure is often faster on modern CPUs because it reduces 
    // the number of operations on the 'res' variable compared to the bottom-up approach.
    for (int i = bit_pos - 1; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}