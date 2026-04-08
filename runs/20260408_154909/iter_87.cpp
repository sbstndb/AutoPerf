#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use trailing zeros to skip initial multiplications where res is just 1.
    // This effectively sets res = base^(2^tz) and reduces exp.
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    
    // Square the base for each trailing zero.
    // This part is strictly sequential.
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }

    uint64_t res = side;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Pre-calculate the number of bits remaining to avoid 'while (exp > 0)' checks.
    // We use __builtin_clzll to find the highest set bit.
    int leading_zeros = __builtin_clzll(exp);
    int bits_to_process = 64 - leading_zeros;

    // Main binary exponentiation loop unrolled to increase ILP.
    // The CPU can execute 'side *= side' (the bottleneck) while 
    // simultaneously calculating 'res *= side' from the previous iteration.
    for (int i = 0; i < bits_to_process; ++i) {
        side *= side;
        if (exp & 1) {
            res *= side;
        }
        exp >>= 1;
    }

    return res;
}