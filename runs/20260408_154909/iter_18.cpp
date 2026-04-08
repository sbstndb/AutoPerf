#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use __builtin_clzll to find the highest set bit.
    // This allows us to skip all leading zeros in the exponent.
    // clzll is undefined for 0, but we handled exp == 0 above.
    int leading_zeros = __builtin_clzll(exp);
    int bit_length = 64 - leading_zeros;

    // Binary Exponentiation (Exponentiation by Squaring)
    // We iterate from the least significant bit up to the highest set bit.
    // To maximize throughput, we use a branchless approach for the result update.
    for (int i = 0; i < bit_length - 1; ++i) {
        // If the bit is set, multiply result by base; otherwise multiply by 1.
        // Modern CPUs execute this pattern efficiently using cmov or bit manipulation.
        uint64_t bit_set = (exp & 1);
        uint64_t multiplier = bit_set ? base : 1;
        result *= multiplier;
        
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the highest bit (which is always 1 at this point)
    return result * base;
}