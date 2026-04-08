#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Use __builtin_ctzll to skip trailing zeros in the exponent
    // This reduces the number of iterations for even exponents.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for the skipped trailing zeros
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    // Binary exponentiation with unrolling and branchless updates
    // We process the remaining bits. Since we handled trailing zeros,
    // the current LSB of exp is guaranteed to be 1.
    while (exp > 1) {
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;

        // Manual unroll to improve ILP
        if (exp > 1) {
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        }
    }

    return result * base;
}