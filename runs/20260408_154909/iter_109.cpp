#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents to bypass bit scanning
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros: reduces iterations by pre-squaring the base.
    // This is highly effective for even exponents.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done.
    if (exp == 1) return base;

    // Binary Exponentiation (Left-to-Right)
    // We start from the second most significant bit because the MSB is always 1.
    uint64_t res = base;
    int bit_pos = 62 - __builtin_clzll(exp);

    // Unrolled loop processing bits from MSB-1 down to 0.
    // Using a branchless approach to avoid pipeline stalls.
    for (; bit_pos >= 0; --bit_pos) {
        res *= res;
        uint64_t bit = (exp >> bit_pos) & 1;
        
        // A branchless selection: if bit is 0, multiply by 1; if 1, multiply by base.
        // Modern x86 compilers optimize this to a CMOV or a simple mask.
        uint64_t multiplier = (bit ? base : 1);
        res *= multiplier;
    }

    return res;
}