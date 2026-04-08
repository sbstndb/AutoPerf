#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^0 = 1, x^0 = 1, 0^x = 0, 1^x = 1
    if (exp == 0) return 1;
    if (base <= 1) return base;

    uint64_t res = 1;

    // Unroll the first few iterations to handle small exponents quickly
    // and reduce the number of jumps.
    
    // Iteration 0
    if (exp & 1) res = base;
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;

    // Iteration 1
    if (exp & 1) res *= base;
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;

    // Main loop: Process 2 bits at a time to increase ILP
    // This allows the CPU to calculate the next base squares 
    // while the result multiplication is pending.
    while (exp > 0) {
        if (exp & 1) res *= base;
        uint64_t next_base = base * base;
        
        if (exp & 2) res *= next_base;
        base = next_base * next_base;
        
        exp >>= 2;
    }

    return res;
}