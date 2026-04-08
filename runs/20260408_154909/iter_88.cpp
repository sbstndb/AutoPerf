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
    
    // If exp was a power of 2, we are done
    if (exp == 1) return side;

    // Determine the highest bit set to bound the loop
    int high_bit = 63 - __builtin_clzll(exp);
    uint64_t res = side;

    // Main binary exponentiation loop
    // We process from the second-highest bit downwards
    for (int i = 1; i <= high_bit; ++i) {
        side *= side;
        
        // Use a lookup table approach to eliminate branching.
        // The compiler will optimize this to a CMOV instruction.
        uint64_t targets[2] = {1, side};
        res *= targets[(exp >> (high_bit - i)) & 1];
    }

    return res;
}