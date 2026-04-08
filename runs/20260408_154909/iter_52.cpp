#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use Bit Scan Reverse to find the highest set bit.
    // This tells us exactly how many squaring steps are needed.
    // __builtin_clzll returns leading zeros; 63 - clz is the index of the MSB.
    int iterations = 63 - __builtin_clzll(exp);

    // We process bits from LSB to MSB-1.
    // The loop is written to encourage the compiler to use CMOV for the result update.
    for (int i = 0; i < iterations; ++i) {
        uint64_t bit = exp & 1;
        // Branchless update: result *= (bit ? base : 1)
        // Compilers typically optimize this to a test + cmov + imul
        if (bit) result *= base;
        
        base *= base;
        exp >>= 1;
    }

    // The final bit (MSB) is always 1, so we do the final multiplication here.
    return result * base;
}