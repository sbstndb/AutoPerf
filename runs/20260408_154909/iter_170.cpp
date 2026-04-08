#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: Handle 0^0 as 1, x^0 as 1, 0^exp as 0, 1^exp as 1
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Special case for base 2: use bit shift. 
    // This is a massive win on x86 and handles the 64-bit overflow correctly.
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation (Square and Multiply)
    // We use a branchless approach for the multiplication to keep the pipeline full.
    // The 'multiplier' is either 'b' (if bit is set) or 1 (if bit is clear).
    while (exp > 1) {
        // If bit is set, multiply res by b, else multiply by 1 (no-op)
        uint64_t mask = (uint64_t)((int64_t)(exp << 63) >> 63);
        res *= (b & mask) | (1ULL & ~mask);
        
        b *= b;
        exp >>= 1;

        // Unroll once to reduce loop overhead and increase ILP
        mask = (uint64_t)((int64_t)(exp << 63) >> 63);
        res *= (b & mask) | (1ULL & ~mask);
        
        b *= b;
        exp >>= 1;
    }

    // Final bit
    if (exp) {
        res *= b;
    }

    return res;
}