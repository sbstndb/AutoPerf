#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Special case: base 0
    if (base == 0) return (exp == 0);
    // Special case: base 1 or exp 0
    if (exp == 0 || base == 1) return 1;
    // Special case: base 2 (optimization for shift)
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    // Find the first set bit to initialize result and avoid one multiplication
    // __builtin_ctzll returns the number of trailing zeros
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for the initial trailing zeros
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Standard binary exponentiation for remaining bits
    // The loop runs log2(exp) times
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}