#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use __builtin_clzll to find the highest set bit.
    // This tells us exactly how many squarings we need.
    int iterations = 63 - __builtin_clzll(exp);

    // Process all bits except the highest one (which is handled by the final multiply)
    // We unroll slightly or use a pattern that avoids the branch.
    for (int i = 0; i < iterations; ++i) {
        // Branchless update of result: 
        // if bit is set, result *= base; else result *= 1;
        uint64_t mask = -(exp & 1);
        result *= (base & mask) | (1ULL & ~mask);
        
        base *= base;
        exp >>= 1;
    }

    // The final result is result * (base^(original_highest_bit))
    return result * base;
}