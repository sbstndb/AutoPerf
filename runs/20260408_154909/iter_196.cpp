#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: exp 0 is 1, base 0/1 are themselves.
    if (exp == 0) return 1;
    if (base <= 1) return base;

    // Remove trailing zeros to reduce the number of iterations.
    // This effectively transforms base^exp into (base^(2^tz))^exp_odd.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;

    // Square the base for each trailing zero.
    // Using a simple loop here as tz is usually very small for 64-bit integers
    // before they overflow to zero.
    for (int i = 0; i < tz; ++i) {
        base *= base;
    }

    uint64_t res = 1;
    
    // Standard Binary Exponentiation (Exponentiation by Squaring)
    // We unroll the loop to improve Instruction Level Parallelism (ILP).
    // The CPU can calculate the next 'base' (squaring) while multiplying 'res'.
    while (exp > 1) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
        
        // Unroll step:
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }

    // Final accumulation
    return res * base;
}