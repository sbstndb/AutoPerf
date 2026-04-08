#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents (0, 1, 2)
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for small bases
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Right-to-Left Binary Exponentiation (Montgomery's Ladder variation)
    // This approach allows for better ILP (Instruction Level Parallelism)
    // as the squaring of the base and the multiplication of the result
    // can be partially overlapped by the CPU's out-of-order execution unit.
    
    // Skip trailing zeros to reduce iterations and avoid starting res at 1
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for every trailing zero
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    if (__builtin_expect(exp == 0, 0)) return res;

    // Main loop: Branchless approach to minimize misprediction penalties
    // Modern CPUs handle the data dependency between base and res efficiently.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}