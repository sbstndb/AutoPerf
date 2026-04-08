#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    uint64_t res = 1;

    // Standard Right-to-Left Binary Exponentiation
    // This structure is highly efficient for ILP as the squaring of 'base'
    // can be calculated independently of the multiplication into 'res'.
    while (true) {
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        base *= base;
    }

    return res;
}