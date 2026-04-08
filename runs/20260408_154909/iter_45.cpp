#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the identity case immediately to allow the loop to assume exp > 0
    if (exp == 0) return 1;
    
    uint64_t result = 1;

    // Use __builtin_ctzll to skip trailing zeros in the exponent.
    // This reduces the number of iterations and handles the initial 'result' assignment.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Instead of starting result at 1 and multiplying, we can skip iterations
    // where the bit is 0. After shifting out trailing zeros, the first bit is 1.
    // We square the base for each trailing zero.
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    
    // The first set bit (originally at trailing_zeros) makes result = base.
    result = base;
    exp >>= 1;

    // Process remaining bits.
    // We use a pattern that encourages the compiler to use IMUL and CMOV
    // to avoid branches inside the loop.
    while (exp > 0) {
        base *= base;
        uint64_t next_res = result * base;
        // If the lowest bit of exp is 1, update result.
        // The compiler optimizes this to a test and cmov.
        if (exp & 1) {
            result = next_res;
        }
        exp >>= 1;
    }

    return result;
}