#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases that would interfere with __builtin_clzll or logic
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case for base 2: shift is significantly faster than multiplication
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Use leading zero count to skip unnecessary iterations.
    // This allows us to start from the most significant bit.
    int leading_zeros = __builtin_clzll(exp);
    
    // Start with the result as the base (implicitly processing the first '1' bit)
    uint64_t res = base;
    
    // Shift exp to align the second most significant bit to the top
    int bits_to_process = 63 - leading_zeros;
    exp <<= (leading_zeros + 1);

    // Main loop: Process bits from MSB to LSB.
    // This structure is often faster than LSB to MSB because it avoids 
    // an extra variable for 'base' squaring in some iterations.
    while (bits_to_process >= 2) {
        // Unroll by 2 to improve ILP and reduce branch overhead
        
        // Bit 1
        res *= res;
        if ((int64_t)exp < 0) res *= base;
        exp <<= 1;

        // Bit 2
        res *= res;
        if ((int64_t)exp < 0) res *= base;
        exp <<= 1;

        bits_to_process -= 2;
    }

    // Handle remaining bit if bits_to_process was odd
    if (bits_to_process) {
        res *= res;
        if ((int64_t)exp < 0) res *= base;
    }

    return res;
}