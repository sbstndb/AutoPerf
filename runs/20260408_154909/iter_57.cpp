#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    
    // Special case for base 2: use bit shift
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Handle even exponents of -1 or other small bases if necessary, 
    // but for uint64_t, we only care about base > 1.

    uint64_t res = 1;
    
    // Binary exponentiation (Square and Multiply)
    // We iterate only up to the highest set bit of exp
    int highest_bit = 63 - __builtin_clzll(exp);
    
    // To reduce the number of multiplications, we start with the 
    // highest bit and work downwards, or start from base and square.
    // Starting from the bottom (LSB) is usually faster for ILP.
    
    while (true) {
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        base *= base;
    }

    return res;
}