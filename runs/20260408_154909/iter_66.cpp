#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    
    // Special case for base 2: use bit shifting
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Optimization: Handle trailing zeros in exponent by squaring the base.
    // This reduces the number of iterations in the main loop.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Manual unrolling of the initial squaring phase
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    // If exp was a power of 2, we are done
    if (exp == 1) return base;

    uint64_t result = 1;

    // Standard Binary Exponentiation (Square-and-Multiply)
    // We use a simple loop because the branch predictor on modern Intel CPUs 
    // is highly efficient at handling the (exp & 1) condition.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Unroll once to improve Instruction Level Parallelism (ILP)
        if (exp > 1) {
            if (exp & 1) {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
    }

    // Final multiplication for the leading bit
    return result * base;
}