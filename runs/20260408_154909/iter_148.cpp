#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Remove trailing zeros to find the first set bit
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Square base for trailing zeros
    // Unrolled loop for common small shift counts
    while (tz >= 2) {
        b *= b;
        b *= b;
        tz -= 2;
    }
    if (tz) b *= b;

    uint64_t res = b;
    exp >>= (__builtin_ctzll(exp) + 1);
    if (exp == 0) return res;

    // Precompute powers for 2-bit window (b^1, b^2, b^3)
    // b^0 is handled by the logic, b^1 is b, b^2 is b2, b^3 is b3
    uint64_t b2 = b * b;
    uint64_t b3 = b2 * b;
    uint64_t table[4] = {0, b, b2, b3};

    // Process 2 bits at a time
    while (exp >= 4) {
        // Square twice for the 2-bit window
        res *= res;
        res *= res;
        
        // Extract 2 bits and multiply
        uint64_t window = exp & 3;
        if (window) {
            res *= table[window];
        }
        
        exp >>= 2;
    }

    // Final bits (1 or 2 bits remaining)
    while (exp > 0) {
        res *= res;
        if (exp & 1) res *= b;
        exp >>= 1;
    }

    return res;
}