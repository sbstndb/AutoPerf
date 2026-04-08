#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^0 = 1, x^0 = 1
    if (exp == 0) return 1;
    // 0^y = 0, 1^y = 1
    if (base <= 1) return base;

    // Use ctz to skip trailing zeros in the exponent.
    // This effectively transforms base^exp into (base^(2^tz))^(exp >> tz)
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Square the base for each trailing zero.
    // This is the "pre-scaling" of the base.
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    // Main Binary Exponentiation loop.
    // We process the remaining bits of the exponent.
    // The loop is structured to minimize the number of operations 
    // on the critical path of 'res'.
    while (exp > 0) {
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
    }

    return res;
}