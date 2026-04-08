#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // 2^exp is just a shift. If exp >= 64, it overflows to 0 for uint64_t.
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Find the first set bit to initialize 'res'
    // This removes the need for 'res = 1' and one iteration of 'res *= base'
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t current_base = base;
    
    // Pre-square current_base to match the first set bit
    for (int i = 0; i < trailing_zeros; ++i) {
        current_base *= current_base;
    }

    uint64_t res = current_base;
    exp >>= (trailing_zeros + 1);

    // Main Square-and-Multiply loop
    // We use a branchless pattern to keep the pipeline full.
    // The compiler will typically generate a TEST + CMOV sequence.
    while (exp > 0) {
        current_base *= current_base;
        
        uint64_t bit = exp & 1;
        uint64_t multiplier = bit ? current_base : 1;
        res *= multiplier;
        
        // Manual unrolling for ILP: process next bit if available
        exp >>= 1;
        if (exp == 0) break;

        current_base *= current_base;
        bit = exp & 1;
        multiplier = bit ? current_base : 1;
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}