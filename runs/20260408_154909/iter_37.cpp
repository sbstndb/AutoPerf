#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Use __builtin_ctzll or clzll to find the highest set bit.
    // This allows us to avoid the 'while (exp > 0)' check and 
    // potentially unroll or use a fixed number of iterations.
    // However, for a general-purpose integer pow, the branchless 
    // update to 'result' is the most significant gain.
    
    uint64_t result = 1;

    // We iterate until exp is 1 to handle the final squaring separately 
    // or avoid it. This reduces the number of imul instructions.
    while (exp > 1) {
        // Branchless update: result = (exp & 1) ? result * base : result;
        // The compiler will typically optimize this to a test + cmov.
        uint64_t next_result = result * base;
        if (exp & 1) result = next_result;
        
        base *= base;
        exp >>= 1;
    }
    
    // The last bit is always 1 at this point (since exp was > 0)
    return result * base;
}