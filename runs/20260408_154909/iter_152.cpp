#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case for base 2: use bit shift
    // This is significantly faster and handles overflow naturally via 64-bit shift rules
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    // Remove trailing zeros from exponent to initialize res
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Square base for trailing zeros
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    // Main loop: Binary Exponentiation (Square and Multiply)
    // We unroll the loop manually to improve ILP (Instruction Level Parallelism)
    // and use a branchless update for 'res'.
    while (exp > 0) {
        b *= b;
        
        // Use a conditional move or mask to update res
        // This avoids branch misprediction penalties
        uint64_t next_res = res * b;
        if (exp & 1) res = next_res;
        
        exp >>= 1;
        
        // Further unrolling for the next bit to reduce loop overhead
        if (exp == 0) break;
        
        b *= b;
        next_res = res * b;
        if (exp & 1) res = next_res;
        
        exp >>= 1;
    }

    return res;
}