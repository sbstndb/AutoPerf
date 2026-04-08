#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and base cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Use trailing zero count to skip initial squarings where res would be 1
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Manual unroll for trailing zeros to reach the first '1' bit
    for (int i = 0; i < (tz & 3); ++i) {
        b *= b;
    }
    for (int i = 0; i < (tz >> 2); ++i) {
        b *= b; b *= b; b *= b; b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Main Square-and-Multiply loop
    // We use a technique to minimize branching. 
    // Modern CPUs handle the multiplication by 1 or b very efficiently.
    while (exp > 0) {
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;

        // Unroll once to reduce loop overhead and provide ILP
        if (exp == 0) break;
        
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
    }

    return res;
}