#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Find the highest set bit to determine the number of iterations
    // This avoids unnecessary loop checks.
    int leading_zeros = __builtin_clzll(exp);
    int iterations = 63 - leading_zeros;
    
    uint64_t res = 1;

    // We process bits from most-significant to least-significant (Square-and-Multiply).
    // This is often faster on modern CPUs as it allows the compiler to 
    // better schedule the independent squaring operations.
    for (int i = iterations; i >= 0; --i) {
        res *= res;
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}