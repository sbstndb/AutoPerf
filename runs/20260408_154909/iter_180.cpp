#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Use ctz to skip trailing zeros in exponent (handles base^(2^n) logic)
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Determine how many bits are left to process to avoid 'while(exp)' branching
    int leading_zeros = __builtin_clzll(exp);
    int bits_left = 64 - leading_zeros;

    // Main loop: Square and Multiply
    // We use a fixed-count loop to allow the compiler to unroll and 
    // the CPU to speculate more effectively.
    for (int i = 0; i < bits_left; ++i) {
        b *= b;
        
        // Use a temporary to allow the compiler to generate a CMOV
        // This avoids branch mispredictions which are costly in tight loops.
        uint64_t next_res = res * b;
        if (exp & 1) {
            res = next_res;
        }
        
        exp >>= 1;
    }

    return res;
}