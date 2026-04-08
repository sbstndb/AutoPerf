#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    // Skip trailing zeros to initialize 'result' with 'base^(2^first_bit)'
    // This avoids the first 'result = 1 * base' multiplication.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base to match the first set bit
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Process remaining bits
    while (exp > 0) {
        base *= base;
        
        // Use a bitwise trick or __builtin_ctzll to skip zeros
        // But for general cases, a simple check with a conditional 
        // multiplication is often fastest on modern x86 due to CMOV.
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}