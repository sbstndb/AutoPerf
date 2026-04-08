#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    // Find the first set bit to initialize 'res' and avoid 
    // multiplying by 1 in the first iteration.
    int ct = __builtin_ctzll(exp);
    uint64_t b = base;
    
    // Square base for the number of trailing zeros
    for (int i = 0; i < ct; ++i) {
        b *= b;
    }
    
    uint64_t res = b;
    exp >>= (ct + 1);

    // Main loop: Binary Exponentiation
    // We use a branchless approach for the multiplication to keep the 
    // pipeline full, as exponent bits are often unpredictable.
    while (exp > 0) {
        b *= b;
        
        // Branchless update: if bit is 0, multiplier is 1. If 1, multiplier is b.
        uint64_t mask = (uint64_t)(-(int64_t)(exp & 1));
        uint64_t multiplier = (b & mask) | (1ULL & ~mask);
        res *= multiplier;
        
        exp >>= 1;
        
        // Small unroll to improve ILP if there are more bits
        if (exp == 0) break;
        
        b *= b;
        mask = (uint64_t)(-(int64_t)(exp & 1));
        multiplier = (b & mask) | (1ULL & ~mask);
        res *= multiplier;
        
        exp >>= 1;
    }

    return res;
}