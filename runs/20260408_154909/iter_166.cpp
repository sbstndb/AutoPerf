#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle small exponents explicitly to avoid loop overhead
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    uint64_t res = 1;
    
    // Standard Binary Exponentiation (Square and Multiply)
    // This is generally faster than windowed methods for 64-bit integers
    // because it avoids memory lookups and complex bit-shifting.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Unroll once to improve Instruction Level Parallelism (ILP)
        if (exp > 1) {
            if (exp & 1) {
                res *= base;
            }
            base *= base;
            exp >>= 1;
        }
    }
    
    return res * base;
}