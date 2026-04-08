#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Handle trailing zeros: base^exp = (base^(2^tz))^(exp >> tz)
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }
    exp >>= tz;
    if (exp == 1) return b;

    // Precompute powers for a 2-bit window: {base^0, base^1, base^2, base^3}
    uint64_t b2 = b * b;
    uint64_t b3 = b2 * b;
    uint64_t powers[4] = {1, b, b2, b3};

    // Find the highest bit to start the windowed approach
    int high_bit = 63 - __builtin_clzll(exp);
    
    // Initialize result with the highest set bit(s)
    // If high_bit is even, we can take 1 bit; if odd, we can take 2.
    // However, to keep it simple and fast, we start with the MSB.
    uint64_t res = b;
    int i = high_bit - 1;

    // Process bits in 2-bit windows where possible
    while (i >= 1) {
        res *= res; // Square
        res *= res; // Square
        
        uint64_t win = (exp >> (i - 1)) & 3;
        res *= powers[win];
        i -= 2;
    }

    // Handle remaining single bit if high_bit was even
    if (i == 0) {
        res *= res;
        if (exp & 1) res *= b;
    }

    return res;
}