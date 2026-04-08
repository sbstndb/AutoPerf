#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Fast path for very small exponents to avoid ctz/shift overhead
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros in exponent
    // (base^exp) = (base^(2^tz))^ (exp >> tz)
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Square base for each trailing zero - unrolled for common small tz
    while (tz >= 2) {
        base *= base;
        base *= base;
        tz -= 2;
    }
    if (tz) base *= base;

    if (exp == 1) return base;

    uint64_t res = 1;
    
    // Unrolled Binary Exponentiation (LSB to MSB)
    // We use a branchless approach for res *= (exp & 1 ? base : 1)
    // to keep the pipeline flowing.
    while (exp > 3) {
        // Bit 0
        uint64_t mask0 = -(exp & 1);
        res *= (base & mask0) | (1 & ~mask0);
        base *= base;
        
        // Bit 1
        uint64_t mask1 = -((exp >> 1) & 1);
        res *= (base & mask1) | (1 & ~mask1);
        base *= base;
        
        exp >>= 2;
    }

    // Handle remaining bits
    if (exp & 1) res *= base;
    if (exp & 2) {
        base *= base;
        res *= base;
    }
    
    return res;
}