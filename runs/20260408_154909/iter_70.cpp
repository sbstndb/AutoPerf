#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle trivial cases immediately
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(base == 0, 0)) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Optimization: Handle small exponents with zero branching
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    uint64_t res = 1;
    
    // Standard Binary Exponentiation (Square and Multiply)
    // We use a simple loop. Modern CPUs like the Ultra 7 255U have 
    // excellent branch prediction. The 'masking' logic in previous 
    // versions often creates longer dependency chains than a simple branch.
    while (exp > 1) {
        if (exp & 1) {
            res *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // Final multiplication for the leading bit
    return res * base;
}