#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common identity cases quickly
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Count leading zeros to skip unnecessary iterations
    // __builtin_clzll is undefined for 0, but we handled exp == 0 above.
    int bits = 64 - __builtin_clzll(exp);
    
    uint64_t result = 1;

    // We iterate through the bits of the exponent.
    // We use a loop that avoids branching on the bits of 'exp'.
    for (int i = 0; i < bits; ++i) {
        // Use a temporary to help the compiler use CMOV instead of a branch
        uint64_t next_result = result * base;
        if (exp & 1) {
            result = next_result;
        }
        
        // Avoid the final squaring of the base on the last bit
        if (i < bits - 1) {
            base *= base;
            exp >>= 1;
        }
    }

    return result;
}