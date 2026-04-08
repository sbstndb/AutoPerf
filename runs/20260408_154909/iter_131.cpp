#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Skip trailing zeros in exp to reduce iterations
    // res = base ^ exp = (base ^ 2^ctz) ^ (exp >> ctz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    if (exp == 0) return res;

    // Binary exponentiation (Right-to-left)
    // We unroll the loop to improve ILP. 
    // The squaring of 'base' and the multiplication of 'res' are independent.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        
        // Manual unroll to help the pipeline
        if (exp == 0) break;
        
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}