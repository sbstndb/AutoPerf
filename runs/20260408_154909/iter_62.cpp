#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    
    // Special case for base 2: shift is much faster than multiplication
    if (base == 2) {
        return (exp >= 64) ? 0 : (1ULL << exp);
    }

    uint64_t res = 1;
    
    // Binary exponentiation unrolled to exploit Instruction Level Parallelism (ILP).
    // By squaring 'base' independently of 'res', the CPU can pipeline the 
    // squaring operations while simultaneously handling the 'res' updates.
    
    // Process bits of exponent
    while (exp > 0) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
        
        if (exp == 0) break;
        
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;

        if (exp == 0) break;

        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;

        if (exp == 0) break;

        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }

    return res;
}