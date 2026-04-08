#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and edge cases
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Find the position of the highest set bit.
    // __builtin_clzll returns the number of leading zeros.
    // For exp = 5 (0...0101), clz is 61, so bit_pos is 2.
    int bit_pos = 63 - __builtin_clzll(exp);
    
    uint64_t result = base;

    // Process from the second-highest bit down to the 0th bit.
    // This avoids the 'result = 1' initialization and the first multiplication.
    for (int i = bit_pos - 1; i >= 0; --i) {
        // Square the current result
        result *= result;
        
        // If the i-th bit of exp is set, multiply by base
        if ((exp >> i) & 1) {
            result *= base;
        }
    }

    return result;
}