#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;

    // Use __builtin_clzll to find the highest set bit and skip leading zeros.
    // This significantly reduces the number of iterations for smaller exponents.
    int leading_zeros = __builtin_clzll(exp);
    
    // We process the bits from most significant to least significant.
    // This is the "Left-to-Right" binary exponentiation algorithm.
    // It is often faster because 'result' starts at 1 and grows, 
    // while 'base' stays constant or is squared.
    
    // Start from the highest bit set
    uint64_t bit = 1ULL << (63 - leading_zeros);
    
    while (bit > 0) {
        result *= result;
        if (exp & bit) {
            result *= base;
        }
        bit >>= 1;
    }

    return result;
}