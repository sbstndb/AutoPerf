#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small bases and exponents
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        return 0; // base is 0
    }

    uint64_t res = 1;
    
    // Right-to-left binary exponentiation (Montgomery's Ladder variation)
    // This approach is generally more pipeline-friendly for the compiler
    // as it separates the squaring of the base from the accumulation into res.
    while (exp > 0) {
        // If the LSB is 1, multiply result by current base
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base for the next bit
        base *= base;
        
        // Shift exponent to process next bit
        exp >>= 1;
        
        // Optimization: if base becomes 0 due to overflow (common in 64-bit), 
        // and exp is still > 0, the result will eventually be 0 or stay same.
        // However, for standard unsigned 64-bit wrap-around, we just continue.
        // The loop will run at most 64 times, but usually much less.
    }

    return res;
}