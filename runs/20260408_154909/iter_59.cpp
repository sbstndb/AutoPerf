#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and identity cases
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(base <= 1, 0)) return base;
    if (__builtin_expect(exp == 1, 0)) return base;

    // Special case for base 2: use bit shift (extremely fast)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Find the highest set bit to determine the number of iterations
    // This allows us to avoid checking 'exp > 0' in every iteration
    int leading_zeros = __builtin_clzll(exp);
    int bits = 63 - leading_zeros;

    uint64_t res = base;
    
    // Process bits from second-highest to lowest
    // We use a local array to eliminate branching for the multiplication
    // The compiler will optimize this into a conditional move (cmov)
    for (int i = bits - 1; i >= 0; --i) {
        res *= res;
        uint64_t choices[2] = {1, base};
        res *= choices[(exp >> i) & 1];
    }

    return res;
}