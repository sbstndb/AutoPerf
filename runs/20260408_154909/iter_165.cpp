#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Use __builtin_clzll to find the most significant bit.
    // This allows us to skip leading zeros and start exactly at the highest power.
    int shift = 63 - __builtin_clzll(exp);
    
    uint64_t res = base;
    
    // Binary exponentiation (Square-and-Multiply)
    // We start from the bit below the MSB because 'res' is initialized to 'base' (base^1)
    for (int i = shift - 1; i >= 0; --i) {
        res *= res; // Square
        
        // If the i-th bit is set, multiply by base
        if ((exp >> i) & 1) {
            res *= base;
        }
    }
    
    return res;
}