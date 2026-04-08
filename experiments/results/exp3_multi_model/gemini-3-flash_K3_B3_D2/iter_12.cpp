#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;

    uint64_t result = 1;

    // Use __builtin_clzll to find the position of the highest set bit.
    // This allows us to avoid unnecessary iterations and squaring.
    int iterations = 64 - __builtin_clzll(exp);

    // Unroll the loop manually to reduce branch overhead and 
    // maximize Instruction Level Parallelism (ILP).
    for (int i = 0; i < iterations; ++i) {
        uint64_t next_result = result * base;
        
        // Use a conditional move (via ternary) to avoid branch misprediction.
        // The compiler translates this to a CMOV instruction.
        result = (exp & 1) ? next_result : result;
        
        // Square the base for the next bit.
        // Note: The last squaring is technically unnecessary but 
        // keeping the loop uniform often helps the CPU's front-end.
        base *= base;
        exp >>= 1;
    }

    return result;
}