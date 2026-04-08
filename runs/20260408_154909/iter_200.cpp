#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Fast path for common small bases
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Right-to-Left Binary Exponentiation
    // This version is generally faster on modern x86 due to better ILP.
    
    // Skip trailing zeros to reduce iterations and initialize 'res'
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    // Note: If exp was power of 2, this handles it efficiently
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    if (exp == 0) return res;

    // Main loop: Unrolled to improve ILP
    // We use a multiplier that is either 'base' or 1 to avoid branching
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        
        // Check if we can exit early to save cycles on small exponents
        if (exp == 0) break;

        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}