#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the edge case immediately. 
    // __builtin_expect tells the compiler this is rare, optimizing code layout.
    if (__builtin_expect(exp == 0, 0)) {
        return 1;
    }
    
    uint64_t result = 1;
    
    // Loop until exp == 1. This saves the final, unnecessary base squaring.
    while (exp > 1) {
        // Compute the next potential result unconditionally.
        uint64_t next_result = result * base;
        
        // Use a ternary operator to encourage a branchless conditional move (cmov).
        result = (exp & 1) ? next_result : result;
        
        // Square the base. This can execute in parallel with the next_result multiplication.
        base *= base;
        exp >>= 1;
    }
    
    // Multiply the remaining base for the final set bit (exp == 1)
    return result * base;
}