#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Handle the common case where exp is small or zero efficiently.
    // Binary exponentiation (Exponentiation by Squaring) reduces complexity 
    // from O(exp) to O(log exp).
    while (exp > 0) {
        // If exponent is odd, multiply result by current base
        if (exp & 1) {
            result *= base;
        }
        // Square the base and halve the exponent
        base *= base;
        exp >>= 1;
    }
    
    return result;
}