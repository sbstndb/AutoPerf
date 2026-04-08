#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents to minimize latency
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Remove trailing zeros to reduce iterations.
    // This ensures the main loop starts with the first set bit.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    
    // Pre-squaring for trailing zeros.
    // Using a simple loop allows the compiler to unroll or use ILP.
    while (tz--) {
        base *= base;
    }

    uint64_t result = base;
    exp >>= 1;

    // Main Binary Exponentiation Loop.
    // We use a branchless approach for the multiplication to avoid 
    // branch misprediction penalties which are costly on x86_64.
    while (exp > 0) {
        base *= base;
        
        // Branchless selection: multiply by base if bit is set, else by 1.
        // Modern compilers optimize this to a test/cmov or a simple mask.
        uint64_t side_effect = (exp & 1) ? base : 1;
        result *= side_effect;
        
        exp >>= 1;
    }

    return result;
}