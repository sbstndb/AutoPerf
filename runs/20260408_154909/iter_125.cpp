#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^0 = 1, x^0 = 1
    if (exp == 0) return 1;
    // 0^x = 0, 1^x = 1
    if (base <= 1) return base;

    uint64_t res = 1;

    // Standard Binary Exponentiation (Square-and-Multiply)
    // Bottom-up approach is generally better for ILP as base*base 
    // can be calculated somewhat independently of the res update.
    while (exp > 1) {
        // If bit is set, multiply result by current base
        // Using a conditional to avoid branch misprediction via compiler optimization
        if (exp & 1) {
            res *= base;
        }
        
        base *= base;
        exp >>= 1;
    }

    // Final multiply for the last remaining bit (MSB is always 1 here)
    return res * base;
}