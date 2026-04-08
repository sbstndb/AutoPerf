#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Fast path for base 2 (Shift)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Fast path for small exponents to avoid loop overhead
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Remove trailing zeros from exponent to initialize 'res'
    // This avoids starting with res = 1 and an extra multiplication
    uint32_t tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Pre-square the base for the trailing zeros
    // Using a simple loop that the compiler can unroll
    for (uint32_t i = 0; i < tz; ++i) {
        base *= base;
    }

    uint64_t res = base;
    exp >>= 1;
    
    // Main Square-and-Multiply loop
    // We use a pattern that maximizes ILP (Instruction Level Parallelism)
    // by calculating the next square independently of the result multiplication.
    while (exp > 0) {
        base *= base;
        
        // Branchless update: if (exp & 1) res *= base;
        // We use a mask to either multiply by 'base' or by 1.
        // This is typically compiled to a test + cmov or a mask + imul.
        uint64_t side_chain = (exp & 1) ? base : 1;
        res *= side_chain;
        
        exp >>= 1;

        // Optional: Early exit if res or base becomes 0 (overflow)
        // In 64-bit unsigned math, base^n becomes 0 very quickly for base > 2
        if (base == 0) break;
    }

    return res;
}