#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Skip trailing zeros in exponent to reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Initial squaring for skipped bits
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main loop: Unrolled to improve ILP and reduce branch pressure
    while (exp > 0) {
        base *= base;
        
        // Use a conditional to avoid branching where possible.
        // The compiler typically optimizes this to a CMOV or 
        // a simple sequence that avoids misprediction penalties.
        if (exp & 1) {
            result *= base;
        }
        
        exp >>= 1;
        
        // Further unrolling to process 2 bits per iteration if possible
        if (exp == 0) break;
        
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}