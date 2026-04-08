#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Standard cases: x^0 = 1
    if (exp == 0) return 1;
    // Optimization for common small exponents
    if (exp == 1) return base;
    if (base <= 1) return base; // 0^n = 0, 1^n = 1 (for n > 0)

    // Use __builtin_clzll to find the highest set bit.
    // This allows us to skip all leading zeros in the exponent.
    // clz is undefined for 0, but we handled exp == 0 above.
    int leading_zeros = __builtin_clzll(exp);
    int bit_width = 63 - leading_zeros;

    uint64_t result = 1;

    // We process the exponent from most significant bit to least significant bit.
    // This is the "Left-to-Right" binary exponentiation algorithm.
    // It is often faster on modern CPUs because the 'result' dependency chain
    // is very clean and it avoids the 'base *= base' calculation on the final bit.
    for (int i = bit_width; i >= 0; --i) {
        result *= result;
        if ((exp >> i) & 1) {
            result *= base;
        }
    }

    return result;
}