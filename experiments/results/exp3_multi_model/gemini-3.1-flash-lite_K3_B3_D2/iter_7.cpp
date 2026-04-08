#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Handle the most common case of small exponents or 0/1 base/exp
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Unroll the loop to process 2 bits per iteration to reduce loop overhead
    // and improve instruction-level parallelism.
    while (exp >= 4) {
        if (exp & 1) result *= base;
        base *= base;
        if (exp & 2) result *= base;
        base *= base;
        exp >>= 2;
    }

    // Handle remaining bits
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    
    return result;
}