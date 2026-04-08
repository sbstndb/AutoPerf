#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common/trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (exp == 1) return base;
    if (base == 1) return 1;

    // Skip trailing zeros to initialize 'res' and reduce iterations.
    // This effectively calculates base^(2^ctz) as the starting point.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square base for the trailing zeros
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;

    // Binary Exponentiation (Exponentiation by Squaring)
    // We use a branchless approach for the result update to maximize throughput.
    while (exp > 0) {
        base *= base;
        
        // Use a conditional move (via ternary) to avoid branch misprediction.
        // The compiler optimizes this to a test/cmov sequence.
        uint64_t multiplier = (exp & 1) ? base : 1;
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}