#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Remove trailing zeros to initialize 'res' and 'base' efficiently
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square base for each trailing zero. 
    // Unrolling this manually as tz is usually small for 64-bit integers.
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Main loop: Process 2 bits at a time to increase ILP and reduce branch pressure
    while (exp >= 2) {
        // Bit 0
        uint64_t b2 = base * base;
        uint64_t m0 = (exp & 1) ? b2 : 1;
        res *= m0;
        
        // Bit 1
        uint64_t b4 = b2 * b2;
        uint64_t m1 = (exp & 2) ? b4 : 1;
        res *= m1;

        base = b4;
        exp >>= 2;
    }

    // Final bit if remaining
    if (exp) {
        base *= base;
        res *= base;
    }

    return res;
}