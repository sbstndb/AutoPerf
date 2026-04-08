#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for identity and zero cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;

    // Use Leading Zero Count to determine the highest bit set.
    // This tells us exactly how many squaring steps are needed.
    int leading_zeros = __builtin_clzll(exp);
    int bit_length = 63 - leading_zeros;

    uint64_t res = 1;
    uint64_t current_base = base;

    // Binary Exponentiation (Square and Multiply)
    // We iterate from the LSB to the MSB. 
    // This structure is highly efficient for modern pipelines.
    for (int i = 0; i < bit_length; ++i) {
        if (exp & 1) {
            res *= current_base;
        }
        current_base *= current_base;
        exp >>= 1;
    }

    // The final bit (MSB) is always 1 because we handled exp=0/1 above.
    // Multiplying at the end avoids one unnecessary squaring of current_base.
    return res * current_base;
}