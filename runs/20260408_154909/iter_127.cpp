#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Skip trailing zeros to reduce iterations
    // If exp = 12 (1100), we can start with base = base^4 and exp = 3 (0011)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    uint64_t res = 1;

    // Unrolled binary exponentiation to improve Instruction Level Parallelism (ILP)
    // We process bits and square the base. By unrolling, the compiler can 
    // overlap the squaring of 'base' with the multiplication into 'res'.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;

        // Manual unroll step
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final bit
    if (exp) {
        res *= base;
    }

    return res;
}