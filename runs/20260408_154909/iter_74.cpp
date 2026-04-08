#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common/trivial cases
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 0) return 0;
        if (base == 1) return 1;
        // base == 2
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Skip trailing zeros in exponent to reduce iterations and initialize 'res'
    // This effectively performs: res = base^(2^ctz); exp >>= ctz;
    int ctz = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < ctz; ++i) {
        b *= b;
    }
    
    uint64_t res = b;
    exp >>= (ctz + 1);

    // LSB-first binary exponentiation
    // This allows the CPU to calculate b *= b and res *= b in parallel
    while (exp > 0) {
        b *= b;
        if (exp & 1) {
            res *= b;
        }
        exp >>= 1;
    }

    return res;
}