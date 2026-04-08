#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small bases and exponents
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
    if (exp == 1) return base;

    uint64_t res = 1;

    // Binary exponentiation (Exponentiation by Squaring)
    // We use a technique to minimize branching. 
    // Most CPUs can optimize the 'if (exp & 1)' into a CMOV, 
    // but we help the compiler by keeping the loop structure tight.
    
    // Peel the first iteration to avoid res = 1 * base
    if (exp & 1) {
        res = base;
    }
    exp >>= 1;

    while (exp > 0) {
        base *= base; // Square the base
        
        if (exp & 1) {
            res *= base; // Multiply result if bit is set
        }
        
        exp >>= 1;

        // Manual unroll to increase Instruction Level Parallelism (ILP)
        // This allows the CPU to calculate the next base*base while 
        // the previous res*base is still in flight.
        if (exp == 0) break;

        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}