#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the 0^0 and x^0 cases immediately.
    if (exp == 0) return 1;

    uint64_t result = 1;

    // Right-to-Left Binary Exponentiation (Square-and-Multiply).
    // This approach is generally more efficient for modern CPUs because
    // it allows for better instruction-level parallelism between 
    // the squaring of the base and the accumulation of the result.
    while (exp > 1) {
        // Use a bitwise mask to eliminate branching.
        // If (exp & 1) is 1, mask is 0xFFFFFFFFFFFFFFFF.
        // If (exp & 1) is 0, mask is 0.
        uint64_t mask = -(exp & 1);
        
        // If the bit is set, multiply result by current base.
        // If not, multiply by 1 (no-op for the value, but keeps pipeline steady).
        // The compiler optimizes (base & mask) | (1 & ~mask) into a CMOV or simple logic.
        result *= (base & mask) | (static_cast<uint64_t>(1) & ~mask);
        
        // Square the base for the next bit.
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the last remaining bit of the exponent.
    // This avoids one unnecessary squaring of 'base' inside the loop.
    return result * base;
}