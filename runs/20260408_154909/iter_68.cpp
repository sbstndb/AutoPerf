#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle trivial cases to return early
    if (__builtin_expect(exp == 0, 0)) return 1;
    if (__builtin_expect(exp == 1, 0)) return base;

    uint64_t result = 1;

    // Standard Binary Exponentiation (Square-and-Multiply)
    // For uint64_t, the loop runs at most 63 times.
    // Branching on (exp & 1) is generally faster than masking because 
    // it avoids unnecessary multiplications and complex bitwise logic.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    // The last bit of exp is always 1 after the loop finishes (since exp > 0)
    // This saves one 'base *= base' calculation.
    return result * base;
}