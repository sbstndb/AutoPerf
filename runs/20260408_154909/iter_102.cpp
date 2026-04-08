#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common identity cases early to reduce latency.
    if (exp == 0) return 1;
    if (base <= 1) return base;

    uint64_t result = 1;

    // Right-to-Left Binary Exponentiation (Square-and-Multiply).
    // This approach is generally faster on modern CPUs because the 
    // update to 'base' (base *= base) and the conditional update to 
    // 'result' can be pipelined more effectively than Left-to-Right.
    while (exp > 1) {
        // If the LSB is set, multiply result by current base.
        // The compiler typically optimizes this into a test and cmov or a short branch.
        if (exp & 1) {
            result *= base;
        }
        
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last remaining bit of the exponent.
    // Doing this outside the loop allows the loop to terminate one iteration earlier.
    return result * base;
}