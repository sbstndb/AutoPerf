#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Base 2 optimization using BMI2/Standard shifts
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation with manual unrolling and branchless logic
    while (exp > 1) {
        // Bit 0
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;

        // Bit 1 (Unrolled to improve ILP)
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;
        
        // The compiler will likely use CMOV for the 'if (exp & 1)' 
        // patterns above when -O3 and -march=native are present.
    }

    // Final bit
    if (exp) res *= b;

    return res;
}