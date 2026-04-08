#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common base cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Use trailing zeros to skip initial multiplications where res would be 1
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Square base for each trailing zero bit
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    // If no bits left, we are done
    if (exp == 0) return res;

    // Determine how many bits are left to process to avoid 'while(exp)' branching
    int leading_zeros = __builtin_clzll(exp);
    int bits_left = 64 - leading_zeros;

    // Main Square-and-Multiply loop
    // We unroll slightly and use a technique that minimizes branching
    for (int i = 0; i < bits_left; ++i) {
        b *= b;
        
        // Use a local array to perform a branchless selection
        // The compiler typically optimizes this to a CMOV or simple mask
        uint64_t choices[2] = {1, b};
        res *= choices[exp & 1];
        
        exp >>= 1;
    }

    return res;
}