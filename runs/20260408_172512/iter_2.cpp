#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle edge case immediately
    if (exp == 0) return 1;
    
    // Peel the first iteration to avoid multiplying by 1
    // and to handle exp == 1 extremely fast.
    uint64_t result = (exp & 1) ? base : 1;
    exp >>= 1;
    if (exp == 0) return result;
    
    base *= base;
    
    // Main exponentiation loop
    while (true) {
        // Compute the next potential result
        uint64_t next = result * base;
        
        // Branchless update using CMOV
        result = (exp & 1) ? next : result;
        
        exp >>= 1;
        
        // Early exit to avoid the final unnecessary squaring
        if (exp == 0) break;
        
        // Square the base (executes in parallel with 'next = result * base' of the NEXT iteration)
        base *= base;
    }
    
    return result;
}