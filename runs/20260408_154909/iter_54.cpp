#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Find the position of the highest set bit.
    // __builtin_clzll is undefined for 0, but we handled exp == 0 above.
    // iterations = 63 - leading_zeros
    int iterations = 63 - __builtin_clzll(exp);
    
    uint64_t result = 1;

    // We process the bits from least significant to most significant.
    // To keep the loop tight and branchless, we use a temporary to 
    // conditionally update the result.
    for (int i = 0; i < iterations; ++i) {
        uint64_t bit = exp & 1;
        // Branchless update: if bit is 0, multiplier is 1; if bit is 1, multiplier is base.
        // This helps the CPU maintain a smooth pipeline.
        uint64_t multiplier = (bit ? base : 1);
        result *= multiplier;
        
        base *= base;
        exp >>= 1;
    }

    // The last bit (the highest one) is always 1 because exp != 0.
    // Multiplying here avoids one iteration of squaring 'base'.
    return result * base;
}