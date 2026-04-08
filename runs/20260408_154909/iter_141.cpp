#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Handle trailing zeros to reduce the exponent magnitude
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }
    exp >>= tz;
    if (exp == 1) return b;

    // Precompute powers for a 2-bit window (base^1, base^2, base^3)
    // This allows us to process 2 bits at a time.
    uint64_t b2 = b * b;
    uint64_t b3 = b2 * b;
    uint64_t lookup[4] = {1, b, b2, b3};

    // Find the highest bit to start processing
    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res = lookup[exp >> (high_bit & ~1)];
    
    // Process 2 bits at a time from top to bottom
    // We adjust the starting point based on whether high_bit is even or odd
    int i = (high_bit & ~1) - 2;
    
    for (; i >= 0; i -= 2) {
        // Square twice for the 2-bit shift
        res *= res;
        res *= res;
        
        // Multiply by precomputed window value
        res *= lookup[(exp >> i) & 3];
    }

    // If there was an odd number of bits, the loop might need a final adjustment
    // but the logic above handles the window alignment via (high_bit & ~1).
    
    return res;
}