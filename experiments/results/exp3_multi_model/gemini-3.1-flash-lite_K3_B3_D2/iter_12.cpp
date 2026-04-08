#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;
    
    // Process 2 bits at a time to reduce loop overhead and branch frequency
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