#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for the most common cases to bypass logic overhead
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(exp == 1, 1)) return base;
    
    // Handle trailing zeros: base = base^(2^tz)
    // This significantly reduces iterations for even exponents.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Unrolled squaring for trailing zeros
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    uint64_t result = base;
    
    // Find the highest set bit to determine loop count
    // This avoids checking 'exp > 0' every iteration
    int leading_zeros = __builtin_clzll(exp);
    int bits = 63 - leading_zeros;
    
    // Main Binary Exponentiation loop
    // We process from the second LSB upwards
    exp >>= 1;
    for (int i = 0; i < bits; ++i) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}