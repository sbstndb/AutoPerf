#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the mathematical identity 0^0 = 1 and n^0 = 1
    if (exp == 0) return 1;
    // Handle 0^n = 0
    if (base == 0) return 0;

    // Find the position of the highest set bit to limit iterations
    int leading_zeros = __builtin_clzll(exp);
    int bit_width = 63 - leading_zeros;

    // Find the first set bit to initialize 'res' and avoid 
    // multiplying by 1 in the first iteration.
    int trailing_zeros = __builtin_ctzll(exp);
    
    // Pre-scale base to the first active bit
    for (int i = 0; i < trailing_zeros; ++i) {
        base *= base;
    }

    uint64_t res = base;
    
    // Process remaining bits from the next bit after trailing_zeros 
    // up to the highest set bit.
    for (int i = trailing_zeros + 1; i <= bit_width; ++i) {
        base *= base;
        // Branchless update: if the i-th bit of exp is set, multiply res by base
        if ((exp >> i) & 1) {
            res *= base;
        }
    }

    return res;
}