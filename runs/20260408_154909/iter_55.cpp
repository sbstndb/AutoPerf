#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use bit scan to find the highest set bit.
    // This tells us exactly how many squaring steps are needed.
    // __builtin_clzll is undefined for 0, but we handled exp=0 above.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;

    // We process the bits from least significant to most significant,
    // but the final bit (the highest one) is handled by the 
    // logic of the loop ending or a final multiplication.
    
    // Unrolling the first iteration to avoid one 'result' update
    if (exp & 1) result = base;
    exp >>= 1;
    
    // If there are no more bits after the shift, we are done.
    if (exp == 0) return result;

    // Main loop: Binary exponentiation (Square and Multiply)
    // We use a fixed number of iterations derived from the exponent's magnitude.
    for (int i = 0; i < iterations; ++i) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}