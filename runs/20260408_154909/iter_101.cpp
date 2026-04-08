#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common/simplest cases immediately.
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;

    // Right-to-Left binary exponentiation (Square-and-Multiply).
    // This approach is generally faster for integer types because it 
    // allows the compiler to better pipeline the squaring of the base 
    // and the conditional multiplication into the result.
    while (exp > 1) {
        // If the current bit is set, multiply the result by the current base.
        // Using a conditional here allows the compiler to use CMOV,
        // avoiding branch mispredictions which are costly in tight loops.
        if (exp & 1) {
            result *= base;
        }
        
        base *= base;
        exp >>= 1;
    }

    // The final multiplication for the last remaining bit (exp is now 1).
    return result * base;
}