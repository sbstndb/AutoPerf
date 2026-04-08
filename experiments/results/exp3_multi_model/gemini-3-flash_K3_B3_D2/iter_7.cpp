#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    // Use __builtin_clzll to find the highest set bit.
    // This tells us exactly how many squaring steps we need.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    uint64_t result = 1;

    // Process bits from least significant to the bit before the most significant.
    // We use a branchless approach to update the result.
    for (int i = 0; i < iterations; ++i) {
        // If the lowest bit is 1, multiply result by current base.
        // If 0, multiply by 1 (no-op for the value, but keeps execution linear).
        uint64_t mask = -(exp & 1); 
        result *= (base & mask) | (1ULL & ~mask);
        
        base *= base;
        exp >>= 1;
    }

    // The final bit (the most significant bit) is always 1 because exp != 0.
    // We multiply the accumulated result by the final base value.
    return result * base;
}