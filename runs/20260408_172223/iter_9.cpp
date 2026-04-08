#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    uint64_t result = 1;
    
    // Unrolling the loop to reduce branch overhead and loop control instructions.
    // We process the bits of the exponent. 
    // Using a conditional move pattern (via multiplication) to avoid branches.
    while (exp > 0) {
        // If exp is odd, multiply result by base.
        // (exp & 1) is 1 if odd, 0 if even.
        // If 1, result = result * base; if 0, result = result * 1.
        uint64_t mask = -(exp & 1);
        result *= (base & mask) | (1 & ~mask);
        
        exp >>= 1;
        if (exp == 0) break;
        
        base *= base;
    }
    
    return result;
}