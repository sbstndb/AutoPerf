#include <cstdint>

/**
 * Optimized integer power function using Exponentiation by Squaring.
 * Complexity: O(log exp) instead of O(exp).
 * This significantly reduces the number of IMUL instructions.
 */
uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;

    // Handle the bits of the exponent from least significant to most significant.
    // For the provided benchmark (max exp 20), this loop runs at most 5 times.
    while (exp > 0) {
        // If the current bit of the exponent is 1, multiply the result by the current base power.
        if (exp & 1) {
            result *= base;
        }
        // Square the base for the next bit position.
        base *= base;
        // Shift exponent to process the next bit.
        exp >>= 1;
    }

    return result;
}