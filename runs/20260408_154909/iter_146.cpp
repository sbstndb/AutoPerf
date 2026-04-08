#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Skip trailing zeros: (base^(2^k))^m = base^(m * 2^k)
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }
    
    exp >>= tz;
    if (exp == 1) return side;

    // Windowed exponentiation (k=2)
    // Precompute window values: base^0, base^1, base^2, base^3
    uint64_t side2 = side * side;
    uint64_t side3 = side2 * side;
    uint64_t lookup[4] = {1, side, side2, side3};

    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res = side;
    int i = high_bit - 2;

    // Process 2 bits at a time
    for (; i >= 0; i -= 2) {
        // Square twice for the 2-bit window
        res *= res;
        res *= res;
        
        // Multiply by precomputed value based on 2 bits
        uint64_t window = (exp >> i) & 3;
        res *= lookup[window];
    }

    // Handle remaining bit if high_bit was odd
    if (i == -1) {
        res *= res;
        if ((exp & 1)) {
            res *= side;
        }
    }

    return res;
}