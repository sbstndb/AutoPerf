#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 0) return 0;
        if (base == 1) return 1;
        // base == 2: Use shift, handle overflow to 0 as per original logic
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // LSB (Least Significant Bit) Square-and-Multiply
    // This approach is generally more efficient for hardware branch predictors
    // and reduces the number of instructions in the hot loop.
    while (exp > 0) {
        // If the current bit is 1, multiply the result by the current base power
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base for the next bit
        base *= base;
        
        // Shift exponent to process the next bit
        exp >>= 1;

        // Optimization: if res becomes 0 due to overflow, we could exit, 
        // but for standard uint64_t wrap-around behavior, we continue.
        // Most integer pow functions assume the user manages overflow.
    }

    return res;
}