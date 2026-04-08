#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the 0^0 and x^0 cases immediately.
    if (exp == 0) return 1;
    
    uint64_t result = 1;

    // Right-to-left binary exponentiation (Square-and-multiply).
    // This approach is generally more efficient for CPUs because it 
    // allows the squaring of the base and the multiplication of the result
    // to be scheduled more effectively by the OOO engine.
    while (exp > 0) {
        // If the current bit of the exponent is 1, multiply result by current base.
        // If 0, multiply by 1 (no-op).
        // The compiler optimizes this into a CMOV (Conditional Move) 
        // which avoids branch misprediction penalties.
        uint64_t multiplier = (exp & 1) ? base : 1;
        result *= multiplier;
        
        // Square the base for the next bit.
        base *= base;
        
        // Shift exponent to process the next bit.
        exp >>= 1;
        
        // Optimization: if exp is 0, we can stop early. 
        // This is faster than calculating the bit-length upfront.
    }

    return result;
}