#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for the most common small exponents
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (base <= 1) return (base == 0) ? 0 : 1;

    uint64_t result = 1;

    // Standard Binary Exponentiation (Right-to-Left)
    // This implementation is optimized for the Intel branch predictor.
    // The 'if (exp & 1)' is typically faster than bit-masking tricks 
    // on modern x86_64 due to macro-op fusion and cmov.
    
    // We iterate until exp is 1 to save one final squaring of the base.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The last bit of exp is always 1 at this point.
    return result * base;
}