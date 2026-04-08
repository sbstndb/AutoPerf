#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp 0 always returns 1
    if (exp == 0) return 1;
    
    // Identity cases: base 0 or 1
    if (base <= 1) return base;

    // Special case: Power of 2
    // Standard C++ unsigned overflow is defined as modulo 2^64.
    // 1ULL << 64 is undefined, so we handle exp >= 64.
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;

    // Bottom-up Binary Exponentiation (Square-and-Multiply)
    // This is generally faster as it reduces the number of iterations 
    // to exactly the position of the highest set bit and simplifies loop control.
    while (exp > 0) {
        // If LSB is set, multiply result by current base
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base for the next bit
        base *= base;
        
        // Shift exponent to process next bit
        exp >>= 1;
        
        // Optimization: if exp is 0, we can stop early before the final base *= base
        // This is particularly effective for small exponents.
        if (exp == 0) break;
    }

    return res;
}