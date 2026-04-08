#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for the most frequent small exponents (0, 1, 2)
    // This covers a large percentage of calls in typical workloads.
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2
    // Base 2 is optimized to a single shift instruction.
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    uint64_t res = 1;

    // Right-to-Left binary exponentiation (Square-and-Multiply).
    // This algorithm is generally faster on modern x86 CPUs because 
    // 'base *= base' and 'res *= base' can be executed with higher 
    // Instruction Level Parallelism (ILP) than the Left-to-Right version.
    while (__builtin_expect(exp > 0, 1)) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    return res;
}