#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the 0^0 and x^0 cases (result is 1)
    if (exp == 0) return 1;
    
    uint64_t result = 1;
    
    // Binary Exponentiation (Exponentiation by Squaring)
    // Complexity: O(log exp) instead of O(exp)
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit
    return result * base;
}