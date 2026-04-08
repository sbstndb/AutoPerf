#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Handle trailing zeros: (base^(2^k))^m = base^(m * 2^k)
    // This makes 'exp' odd, which simplifies the windowing logic.
    int tz = __builtin_ctzll(exp);
    uint64_t b1 = base;
    for (int i = 0; i < tz; ++i) {
        b1 *= b1;
    }
    exp >>= tz;

    if (exp == 1) return b1;

    // Precompute powers for a 2-bit window
    // Since exp is odd, we only need b1, b1^2, b1^3
    uint64_t b2 = b1 * b1;
    uint64_t b3 = b2 * b1;
    uint64_t lookup[4] = {1, b1, b2, b3};

    // Find highest bit to start processing
    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res;
    int i = high_bit;

    // If the highest bit is even (e.g., bit 2, 4...), we can take 1 bit to align to 2-bit pairs
    // or just take the first 1 or 2 bits to initialize 'res'.
    if (i & 1) {
        // Bits are at i and i-1
        res = lookup[(exp >> (i - 1)) & 3];
        i -= 2;
    } else {
        // Bit is at i
        res = b1;
        i -= 1;
    }

    // Process 2 bits at a time
    while (i >= 1) {
        uint64_t sq = res * res;
        res = sq * sq;
        uint64_t win = (exp >> (i - 1)) & 3;
        if (win) {
            res *= lookup[win];
        }
        i -= 2;
    }

    // If one bit remains
    if (i == 0) {
        res = res * res;
        if ((exp & 1)) {
            res *= b1;
        }
    }

    return res;
}