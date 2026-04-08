#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Trivial cases handled with minimal branching
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    
    // Special case for base 2: use shift if within range, else 0 (overflow)
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Binary Exponentiation (Right-to-Left)
    // We use __builtin_ctzll and __builtin_clzll to minimize iterations.
    // However, for maximum ILP and to avoid branch mispredictions, 
    // we unroll the loop logic.
    
    uint64_t res = 1;

    // Handle trailing zeros in exp to reduce work
    // exp = odd_part * 2^tz. So base^exp = (base^(2^tz))^odd_part
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // Now exp is odd. We process bits from LSB to MSB.
    // We unroll the loop to allow the compiler to schedule imul instructions better.
    // Since uint64_t max is 2^64, and base >= 3, exp cannot be very large 
    // without overflowing, but we support the full 64-bit range.
    
    while (exp > 0) {
        if (exp & 1) {
            res *= base;
        }
        
        // If exp is 1, we are done, avoid the last squaring
        if (exp == 1) break;
        
        base *= base;
        exp >>= 1;

        // Manual unroll to improve ILP
        if (exp & 1) {
            res *= base;
        }
        if (exp == 1) break;
        
        base *= base;
        exp >>= 1;
    }

    return res;
}