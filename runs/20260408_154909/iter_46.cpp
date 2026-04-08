#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    // Skip trailing zeros in exponent to initialize result
    // result = base^(2^ctz(exp))
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t res_base = base;
    for (int i = 0; i < trailing_zeros; ++i) {
        res_base *= res_base;
    }
    
    uint64_t result = res_base;
    exp >>= (trailing_zeros + 1);

    // Process remaining bits
    while (exp > 0) {
        res_base *= res_base;
        
        // Use a temporary to allow the compiler to generate a CMOV
        // instead of a branch for the conditional multiplication.
        uint64_t side_product = result * res_base;
        if (exp & 1) {
            result = side_product;
        }
        
        exp >>= 1;
    }

    return result;
}