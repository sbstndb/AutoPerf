#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle base cases immediately
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Fast path for very small exponents
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;

    // Use __builtin_ctzll to find the first set bit and initialize res.
    // This removes the need for 'res = 1' and the first few multiplications.
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < trailing_zeros; ++i) {
        b *= b;
    }
    
    uint64_t res = b;
    exp >>= (trailing_zeros + 1);

    if (exp == 0) return res;

    // Binary Exponentiation with reduced branching.
    // We use the leading bit count to determine exactly how many iterations remain.
    int remaining_bits = 63 - __builtin_clzll(exp);
    
    for (int i = 0; i <= remaining_bits; ++i) {
        b *= b;
        // Using a temporary and a conditional to encourage the compiler 
        // to use a CMOV (conditional move) instead of a branch.
        uint64_t next_res = res * b;
        if (exp & 1) {
            res = next_res;
        }
        exp >>= 1;
    }

    return res;
}