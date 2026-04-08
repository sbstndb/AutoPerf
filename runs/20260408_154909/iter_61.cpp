#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    
    // Special case for base 2: use bit shifting
    // Note: 2^64 and above will overflow to 0 in 64-bit math, 
    // which matches the behavior of repeated multiplication.
    if (base == 2) {
        return (exp >= 64) ? 0 : (1ULL << exp);
    }

    // Skip trailing zeros in exponent to reduce iterations
    // res = base^(2^ctz)
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    
    uint64_t res = base;
    while (ctz--) {
        res *= res;
    }

    // If exp was a power of 2, we are done
    if (exp == 1) return res;

    // Standard binary exponentiation for remaining bits
    // We start from the second bit because res is already base^(2^ctz)
    uint64_t accumulator = res;
    uint64_t current_sq = res;
    exp >>= 1;

    while (exp > 0) {
        current_sq *= current_sq;
        if (exp & 1) {
            accumulator *= current_sq;
        }
        exp >>= 1;
    }

    return accumulator;
}