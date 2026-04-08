#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common small exponents immediately to avoid loop overhead
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;
    
    // Binary Exponentiation (Square and Multiply)
    // This reduces complexity from O(exp) to O(log exp)
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // The final multiplication is done here because the last 'base *= base' 
    // in the loop is always redundant when exp reaches 1.
    return result * base;
}