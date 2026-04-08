#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small exponents to bypass logic overhead
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Handle trailing zeros: base = base^(2^tz)
    // This is mathematically equivalent to shifting the exponent,
    // but doing it upfront simplifies the main loop logic.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    // After shifting out trailing zeros, the LSB is 1.
    // We start 'result' at 'base' and move to the next bit.
    uint64_t result = base;
    exp >>= 1;
    
    if (exp == 0) return result;

    // Main Square-and-Multiply loop.
    // We use a bit-scan to find the highest set bit to limit iterations,
    // but for x86_64, a simple 'while(exp)' with a branchless update 
    // often performs best due to the internal 'cmov' optimization.
    // We use a local array to eliminate the branch for 'result *= base'.
    while (exp > 0) {
        base *= base;
        uint64_t side_effect[2] = {1, base};
        result *= side_effect[exp & 1];
        exp >>= 1;
    }

    return result;
}