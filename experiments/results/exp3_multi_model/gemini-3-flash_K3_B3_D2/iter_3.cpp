#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    uint64_t result = 1;
    
    // Standard Exponentiation by Squaring (O(log exp))
    // This is significantly faster than the O(exp) linear loop.
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Optimization: if base becomes 0 due to overflow in 64-bit math,
        // and result is already updated, further iterations are useless.
        // However, in standard C++ unsigned overflow is defined as wrap-around,
        // so we continue to ensure identical results to the original loop.
    }
    
    return result;
}