#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Skip trailing zeros in exponent to reduce iterations
    // and initialize result with the first required power of base.
    int trailing_zeros = __builtin_ctzll(exp);
    uint64_t b = base;
    for (int i = 0; i < trailing_zeros; ++i) {
        b *= b;
    }
    
    uint64_t result = b;
    exp >>= (trailing_zeros + 1);

    // Main loop: Process remaining bits
    // We use a branchless approach where possible to maximize throughput
    while (exp > 0) {
        b *= b;
        if (exp & 1) {
            result *= b;
        }
        exp >>= 1;
        
        // Manual unroll for the next bit to improve ILP
        if (exp == 0) break;
        
        b *= b;
        if (exp & 1) {
            result *= b;
        }
        exp >>= 1;
    }
    
    return result;
}