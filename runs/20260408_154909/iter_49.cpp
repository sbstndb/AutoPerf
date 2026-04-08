#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Find the position of the highest set bit to avoid unnecessary iterations.
    // __builtin_clzll returns the number of leading zeros.
    // For exp = 5 (101 binary), leading zeros for 64-bit is 61.
    // bits = 64 - 61 = 3.
    int bits = 64 - __builtin_clzll(exp);
    
    uint64_t result = 1;

    // We iterate from the highest bit down to the lowest.
    // This "Left-to-Right" binary exponentiation is often faster 
    // because it keeps the 'result' variable as the primary accumulator.
    for (int i = bits - 1; i >= 0; --i) {
        result *= result;
        
        // Use a temporary to allow the compiler to generate a CMOV (Conditional Move)
        // instead of a branch. This prevents branch misprediction penalties.
        uint64_t multiplier = (exp >> i) & 1;
        if (multiplier) {
            result *= base;
        }
    }

    return result;
}