#include <cstdint>

/**
 * Optimized integer power function.
 * Uses a branchless approach to Exponentiation by Squaring to avoid 
 * branch misprediction penalties which are dominant for small exponents.
 */
uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;

    uint64_t res = 1;

    // Bit 0
    uint64_t bit_val = (exp & 1) ? base : 1;
    res *= bit_val;
    
    // Bit 1
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;
    bit_val = (exp & 1) ? base : 1;
    res *= bit_val;

    // Bit 2
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;
    bit_val = (exp & 1) ? base : 1;
    res *= bit_val;

    // Bit 3
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;
    bit_val = (exp & 1) ? base : 1;
    res *= bit_val;

    // Bit 4 (Handles up to exp 31)
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;
    bit_val = (exp & 1) ? base : 1;
    res *= bit_val;

    // Final check for larger exponents (though benchmark max is 20)
    exp >>= 1;
    while (exp > 0) {
        base *= base;
        if (exp & 1) res *= base;
        exp >>= 1;
    }

    return res;
}