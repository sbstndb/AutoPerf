#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for small exponents - extremely common in many workloads
    if (__builtin_expect(exp <= 2, 1)) {
        if (exp == 2) return base * base;
        if (exp == 1) return base;
        if (exp == 0) return 1;
    }

    // Special case for base 0, 1, 2
    if (__builtin_expect(base <= 2, 0)) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        if (base == 1) return 1;
        if (base == 0) return 0;
    }

    // Use bit scan to skip leading zeros of the exponent
    // This reduces the number of iterations significantly for small/medium exponents
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t res = 1;
    
    // We start from the most significant bit and move downwards.
    // This "Left-to-Right" binary exponentiation is often faster on modern CPUs
    // because the 'res' dependency chain is cleaner.
    
    // First bit is always 1 (since we handled exp=0 and found clz), 
    // so we can initialize res to base and start from the next bit.
    res = base;
    bit_pos--;

    // Unrolled loop to process bits
    while (bit_pos >= 0) {
        res *= res; // Square
        if ((exp >> bit_pos) & 1) {
            res *= base; // Multiply
        }
        bit_pos--;
        
        // Manual unroll for performance
        if (bit_pos < 0) break;
        
        res *= res;
        if ((exp >> bit_pos) & 1) {
            res *= base;
        }
        bit_pos--;
    }

    return res;
}