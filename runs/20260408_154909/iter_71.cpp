#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle common base cases and edge cases early
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 0) return 0;
        if (base == 1) return 1;
        // base == 2: 1ULL << 64 is undefined behavior, handle via comparison
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // Use the Right-to-Left (LSB) binary exponentiation algorithm.
    // This is generally faster as it allows the CPU to overlap the 
    // squaring of the base and the multiplication into the result.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last bit (exp is now 1)
    // This avoids one unnecessary 'base *= base' operation.
    return res * base;
}