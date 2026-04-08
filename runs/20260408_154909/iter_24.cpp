#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // If base is 2, we can use a shift. Note: result > 64 bits will wrap (standard behavior)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Find the position of the highest set bit to limit iterations
    int leading_zeros = __builtin_clzll(exp);
    int bits = 63 - leading_zeros;

    // Start with the result as the base
    uint64_t res = base;

    // Top-down Square-and-Multiply (Montgomery Ladder style)
    // This approach is often faster on modern CPUs because it reduces 
    // the number of operations compared to the trailing-zero skip method
    // for many common exponent distributions.
    for (int i = bits - 1; i >= 0; --i) {
        res *= res;
        
        // Branchless multiplication:
        // If the i-th bit is set, multiply by base.
        // Using a small array or a ternary often compiles to a CMOV or 
        // simple arithmetic, avoiding branch mispredictions.
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}