#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2 (Standard wrap-around behavior)
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Bottom-up Binary Exponentiation (Exponentiation by Squaring)
    // This version is more ILP-friendly than the top-down approach.
    
    // Optimization: Skip trailing zeros to initialize 'res'
    // This reduces the number of multiplications and iterations.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Pre-square the base for the skipped bits
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = 1;
    
    // Main loop: Process remaining bits
    // We use a pattern that encourages the compiler to use CMOV
    while (exp > 0) {
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break; // Avoid unnecessary last multiplication of base
        base *= base;
    }

    return res;
}