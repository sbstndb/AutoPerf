#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base 0 case immediately
    if (base == 0) return exp == 0 ? 1 : 0;
    // Handle exp 0 case (also covers 0^0 if logic reached here)
    if (exp == 0) return 1;

    // Skip trailing zeros in exponent to initialize result
    // This reduces the number of iterations and avoids result = 1 * base
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for each trailing zero
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Standard binary exponentiation for remaining bits
    // The compiler can often unroll this or use cmov for the multiplication
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}