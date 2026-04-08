#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for common small cases to bypass logic overhead
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Skip trailing zeros to initialize 'res' with a squared version of 'base'
    // This reduces the total number of multiplications in the main loop.
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    
    // Initial squaring for trailing zeros
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }

    uint64_t res = side;
    exp >>= (tz + 1);

    if (exp == 0) return res;

    // Determine the number of bits remaining to process
    // This allows the compiler to optimize the loop structure.
    int leading_bits = 64 - __builtin_clzll(exp);

    // Main Loop: Square and Multiply
    // We use a lookup-style multiplication to avoid branching on 'exp & 1'
    for (int i = 0; i < leading_bits; ++i) {
        side *= side;
        
        // Use a temporary array to eliminate branching.
        // The compiler will typically optimize this to a CMOV or 
        // simple dependency chain, preventing pipeline stalls.
        uint64_t vals[2] = {1, side};
        res *= vals[exp & 1];
        
        exp >>= 1;
    }

    return res;
}