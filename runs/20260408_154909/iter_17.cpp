#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common small exponents immediately to avoid loop overhead
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base <= 1) return (base == 0) ? 0 : 1;

    // Skip trailing zeros in exponent to reduce iterations
    // Example: base^12 = (base^4)^3. 
    // This reduces the number of times we update 'result'.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    while (trailing_zeros--) {
        base *= base;
    }

    uint64_t result = 1;

    // Main loop: Process bits of exponent
    // We use a pattern that encourages the compiler to use CMOV or 
    // minimize branch penalties.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;

        // Manual unroll to improve ILP (Instruction Level Parallelism)
        if (exp <= 1) break;
        
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    return result * base;
}