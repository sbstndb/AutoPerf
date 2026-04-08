#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: handle early to minimize latency for trivial calls
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // Optimized for x86_64 using SHLX/CMOV logic
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Bottom-up Square-and-Multiply
    // This allows the CPU to calculate base^2, base^4, etc., in parallel 
    // with the result accumulation, increasing Instruction Level Parallelism (ILP).
    
    // Skip trailing zeros to initialize 'res' and reduce iterations
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Pre-square the base for the first set bit
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }
    
    uint64_t res = base;
    exp >>= 1;

    // Process remaining bits
    while (exp > 0) {
        base *= base;
        
        // Use a branchless update for the result.
        // The compiler typically generates a CMOV or a sequence of 
        // instructions that avoid branch misprediction penalties.
        uint64_t multiplier = (exp & 1) ? base : 1;
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}