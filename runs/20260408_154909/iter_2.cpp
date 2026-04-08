#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Use Exponentiation by Squaring (Binary Exponentiation)
    // Complexity: O(log exp) instead of O(exp)
    while (exp > 0) {
        // If exponent is odd, multiply result by current base
        if (exp & 1) {
            result *= base;
        }
        // Square the base and shift exponent right
        base *= base;
        exp >>= 1;
    }
    
    return result;
}