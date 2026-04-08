#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // Standard behavior for 64-bit unsigned: 2^64 is 0 (wrap around)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Skip trailing zeros in exponent to initialize 'res'
    // This reduces the number of iterations and avoids res = 1 initialization
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Pre-square base for the skipped bits
    // This part is usually very fast as trailing_zeros is often small
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Bottom-up binary exponentiation (Square-and-Multiply)
    // This is more ILP-friendly than the top-down approach
    while (exp > 0) {
        base *= base;
        
        // Use a conditional move (via ternary) to avoid branch mispredictions
        // The compiler optimizes this to: test/cmov or similar non-branching logic
        uint64_t multiplier = (exp & 1) ? base : 1;
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}