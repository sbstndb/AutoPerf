#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the zero base case as a cold path
    if (__builtin_expect(base == 0, 0)) {
        return (exp == 0) ? 1 : 0;
    }

    // Handle the zero exponent case
    if (exp == 0) return 1;

    uint64_t result = 1;

    // Binary Exponentiation (Exponentiation by Squaring)
    // We unroll the first check to avoid result = 1 * base
    // and use a branchless approach for the remaining bits.
    while (true) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        base *= base;
    }

    return result;
}