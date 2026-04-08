#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases
    if (exp == 0) return 1;
    if (base <= 1) return base;

    uint64_t result = 1;
    
    // Skip leading zeros to reduce iterations
    int shift = __builtin_ctzll(exp);
    exp >>= shift;
    
    // Square the base for the skipped zeros
    while (shift--) {
        base *= base;
    }

    // Process remaining bits
    // Unrolling slightly to help the compiler pipeline the imul instructions
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        base *= base;
    }
    
    return result;
}