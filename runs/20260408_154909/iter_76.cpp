#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common bases and exponents
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 0) return 0;
        if (base == 1) return 1;
        // base == 2: Use shift, handle overflow to 0 as per original logic
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    uint64_t res = 1;
    
    // Right-to-Left Binary Exponentiation (Montgomery's Ladder variation)
    // This approach increases ILP as 'base *= base' and 'res *= base' 
    // can be partially overlapped by the CPU's out-of-order engine.
    
    // Optimization: Skip trailing zeros in exponent to reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for all skipped trailing zero bits
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    // Main loop: Process remaining bits
    while (exp > 0) {
        // If the current bit is 1, multiply result by current base
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base for the next bit
        base *= base;
        exp >>= 1;
        
        // Note: The branch 'if (exp & 1)' is generally well-predicted 
        // for small exponents or handled via CMOV by modern compilers.
    }

    return res;
}