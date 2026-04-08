#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Skip trailing zeros in exponent to reduce iterations
    // Example: base^12 = (base^4)^3. 
    // This reduces the number of times we update 'result'.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    while (trailing_zeros--) {
        base *= base;
    }

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Exponentiation by squaring
    // We use a pattern that encourages the compiler to use 
    // conditional moves (cmov) or keep the pipeline full.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}