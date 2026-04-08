#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Bottom-up binary exponentiation (Square and Multiply)
    // This approach is often more ILP-friendly than top-down.
    
    // Skip trailing zeros in the exponent to reduce iterations
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square the base for the initial trailing zeros
    // We do this separately to initialize 'res' without a branch
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Main loop: Process remaining bits
    // We use a branchless approach to avoid misprediction penalties
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}