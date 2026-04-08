#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases quickly
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Skip trailing zeros to initialize 'res' with a power of 'base'
    // This avoids multiplying res by 1 in the first few iterations
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }

    uint64_t res = side;
    exp >>= (tz + 1);

    // Main loop: Square and Multiply
    // Unrolled to process 2 bits at a time to improve ILP and reduce branch overhead
    while (exp > 0) {
        // Bit 0
        side *= side;
        if (exp & 1) res *= side;
        exp >>= 1;

        if (exp == 0) break;

        // Bit 1
        side *= side;
        if (exp & 1) res *= side;
        exp >>= 1;
    }

    return res;
}