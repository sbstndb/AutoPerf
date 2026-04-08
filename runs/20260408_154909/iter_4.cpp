#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Exponentiation by squaring (Binary Exponentiation)
    // Complexity: O(log exp) instead of O(exp)
    while (exp > 0) {
        // If the current bit of the exponent is 1, multiply the result by the current base
        if (exp & 1) {
            result *= base;
        }
        // Square the base for the next bit
        base *= base;
        // Shift the exponent to process the next bit
        exp >>= 1;
    }
    
    return result;
}