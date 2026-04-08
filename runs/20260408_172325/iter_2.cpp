#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Exponentiation by squaring: O(log exp) complexity
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