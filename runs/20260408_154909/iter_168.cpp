#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use __builtin_clzll to skip leading zeros and find the highest set bit.
    // We start 'res' with 'base' and process bits from high_bit - 1 down to 0.
    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res = base;

    // Main Square-and-Multiply loop
    // This is generally faster than windowed approaches for 64-bit integers
    // because it avoids stack-based lookup tables and complex bit-shifting.
    for (int i = high_bit - 1; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}