#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2 (common in alignment/bit manipulation)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Handle trailing zeros in exponent to reduce iterations
    // This effectively transforms base^exp into (base^(2^tz))^ (exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square base 'tz' times
    while (tz--) {
        base *= base;
    }

    // Fast path for small exponents after shifting
    if (exp == 1) return base;

    uint64_t res = base;
    
    // Main Binary Exponentiation Loop (Square-and-Multiply)
    // Unrolled to improve Instruction Level Parallelism (ILP)
    while (exp > 1) {
        exp >>= 1;
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        
        // Second unroll step if still bits remaining
        if (exp <= 1) break;
        
        exp >>= 1;
        base *= base;
        if (exp & 1) {
            res *= base;
        }
    }

    return res;
}