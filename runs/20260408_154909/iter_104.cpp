#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the identity case immediately.
    if (exp == 0) return 1;
    if (base == 0) return 0;

    uint64_t result = 1;

    // Right-to-Left Binary Exponentiation (Square-and-Multiply).
    // This approach is generally more efficient for modern CPUs because
    // it allows the compiler to better utilize the execution units.
    while (exp > 0) {
        // Use a conditional move (CMOV) logic via the ternary operator.
        // This prevents branch mispredictions which are costly in tight loops.
        // The compiler will generate: test/and, cmov, imul.
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;

        // Square the base for the next bit.
        base *= base;
        
        // Shift the exponent.
        exp >>= 1;

        // Optimization: If exp is 0, we can exit early. 
        // This is particularly effective for small exponents.
        if (exp == 0) break;
    }

    return result;
}