#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Unroll the loop to process bits in chunks. 
    // This reduces branch overhead and allows for better instruction scheduling.
    while (exp > 1) {
        // Handle the current bit: result *= (exp & 1) ? base : 1
        // Using a temporary multiplier to encourage the compiler to use CMOV
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;
        
        // Square the base for the next bit
        base *= base;
        exp >>= 1;

        // Manual unroll of the second bit to increase ILP
        if (exp > 1) {
            uint64_t multiplier2 = (exp & 1) ? base : 1;
            result *= multiplier2;
            base *= base;
            exp >>= 1;
        }
    }

    // Final bit processing
    return result * base;
}