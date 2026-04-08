#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Fast path for trailing zeros (even numbers and powers of 2)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }
    
    uint64_t result = 1;
    
    // 8-way unrolled branchless binary exponentiation
    while (exp > 0) {
        uint64_t mult1 = (exp & 1) ? base : 1ULL;
        result *= mult1;
        if (exp < 2) break;
        base *= base;

        uint64_t mult2 = (exp & 2) ? base : 1ULL;
        result *= mult2;
        if (exp < 4) break;
        base *= base;

        uint64_t mult3 = (exp & 4) ? base : 1ULL;
        result *= mult3;
        if (exp < 8) break;
        base *= base;

        uint64_t mult4 = (exp & 8) ? base : 1ULL;
        result *= mult4;
        if (exp < 16) break;
        base *= base;

        uint64_t mult5 = (exp & 16) ? base : 1ULL;
        result *= mult5;
        if (exp < 32) break;
        base *= base;

        uint64_t mult6 = (exp & 32) ? base : 1ULL;
        result *= mult6;
        if (exp < 64) break;
        base *= base;

        uint64_t mult7 = (exp & 64) ? base : 1ULL;
        result *= mult7;
        if (exp < 128) break;
        base *= base;

        uint64_t mult8 = (exp & 128) ? base : 1ULL;
        result *= mult8;
        
        exp >>= 8;
        if (!exp) break;
        base *= base;
    }
    
    return result;
}