#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Handle trailing zeros to reduce 'exp'
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }
    exp >>= tz;

    if (exp == 1) return b;

    // Precompute powers for a 2-bit window
    uint64_t b2 = b * b;
    uint64_t b3 = b2 * b;
    uint64_t lookup[4] = {1, b, b2, b3};

    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res;
    int i = high_bit;

    // Initialize res based on the top 1 or 2 bits
    if (i & 1) {
        // Even number of bits remaining (e.g., 10xx or 11xx)
        res = lookup[exp >> (i - 1)];
        i -= 2;
    } else {
        // Odd number of bits remaining (e.g., 1xx)
        res = b;
        i -= 1;
    }

    // Process 2 bits at a time
    while (i >= 1) {
        // Square twice: res = res^4
        res *= res;
        res *= res;
        
        // Multiply by base^window
        res *= lookup[(exp >> (i - 1)) & 3];
        i -= 2;
    }

    // Handle remaining bit if necessary
    if (i == 0) {
        res = res * res * lookup[exp & 1];
    }

    return res;
}