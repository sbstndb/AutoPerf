#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;
    
    // Use __builtin_clzll to find the highest set bit.
    // This tells us exactly how many squaring steps are needed.
    // clzll is undefined for 0, but we handled exp == 0 above.
    int iterations = 63 - __builtin_clzll(exp);

    // Binary exponentiation (Square and Multiply)
    // We iterate through the bits of the exponent.
    for (int i = 0; i < iterations; ++i) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The final result is the accumulated product times the final base power.
    // This structure helps the compiler optimize the loop exit and 
    // reduces the total number of multiplications by one compared to a naive loop.
    return result * base;
}