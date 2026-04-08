#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Use trailing zeros to skip initial multiplications
    int tz = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Squaring loop for trailing zeros
    // This is highly predictable and low latency
    for (int i = 0; i < tz; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Main Square-and-Multiply loop
    // We unroll to process 2 bits at a time to increase ILP.
    // This allows the CPU to calculate b^2 and b^4 while 
    // simultaneously deciding whether to multiply res by them.
    while (exp >= 4) {
        // Bit 0
        b *= b;
        if (exp & 1) res *= b;
        
        // Bit 1
        b *= b;
        if (exp & 2) res *= b;
        
        exp >>= 2;
    }

    // Final bits
    while (exp > 0) {
        b *= b;
        if (exp & 1) res *= b;
        exp >>= 1;
    }

    return res;
}