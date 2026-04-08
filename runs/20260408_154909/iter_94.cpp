#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Power of 2 optimization: use SHL
    // Note: 1ULL << 64 is undefined behavior, so we handle exp >= 64
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // Bottom-up Binary Exponentiation (Exponentiation by Squaring)
    // This version is more ILP-friendly than the top-down approach.
    // We use a while loop that the compiler can unroll.
    while (exp > 1) {
        // If LSB is 1, multiply res by current base
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base
        base *= base;
        
        // Shift exponent
        exp >>= 1;
        
        // Manual unroll hint: handle the next bit to reduce branch overhead
        if (exp <= 1) break;
        
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit
    if (exp) {
        res *= base;
    }

    return res;
}