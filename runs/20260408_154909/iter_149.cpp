#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Use a 2-bit window (sliding window/k-ary exponentiation)
    // Precompute base^1, base^2, base^3
    uint64_t b2 = base * base;
    uint64_t b3 = b2 * base;
    uint64_t lookup[4] = {1, base, b2, b3};

    // Find the highest set bit to start from the top
    // This avoids the "trailing zeros" logic and handles the first bits more cleanly
    int leading_zeros = __builtin_clzll(exp);
    int bits_left = 64 - leading_zeros;

    // Initialize result with the topmost bits
    // If bits_left is odd, we take 1 bit; if even, we take 2 bits to keep window aligned
    uint64_t res;
    if (bits_left & 1) {
        res = base;
        bits_left -= 1;
        exp &= ~(1ULL << (63 - leading_zeros));
    } else {
        int shift = bits_left - 2;
        res = lookup[(exp >> shift) & 3];
        bits_left -= 2;
    }

    // Process 2 bits at a time
    while (bits_left > 0) {
        bits_left -= 2;
        
        // Square twice for the 2-bit window
        res *= res;
        res *= res;

        // Multiply by precomputed base^window
        uint64_t window = (exp >> bits_left) & 3;
        if (window > 0) {
            res *= lookup[window];
        }
    }

    return res;
}