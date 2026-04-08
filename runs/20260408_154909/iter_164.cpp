#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Remove trailing zeros from exponent to reduce iterations
    // base^exp = (base^(2^tz))^(exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square the base for each trailing zero
    // This is more efficient than a loop for small tz
    while (tz--) {
        base *= base;
    }

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    // Standard Right-to-Left Binary Exponentiation
    // This approach allows for better ILP than Left-to-Right
    uint64_t res = 1;
    
    // First iteration outside to avoid res = 1 * base
    if (exp & 1) {
        res = base;
    }
    
    while (exp > 1) {
        exp >>= 1;
        base *= base; // Square the base
        if (exp & 1) {
            res *= base; // Multiply result by current base power
        }
    }

    return res;
}