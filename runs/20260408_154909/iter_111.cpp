#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for the most common cases
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;
    
    // Standard Binary Exponentiation (Square-and-Multiply)
    // This structure is highly efficient and avoids complex branching
    // that can cause compiler optimization failures.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Final multiplication for the last remaining bit
    return result * base;
}