#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
    if (exp == 1) return base;

    uint64_t res = 1;
    uint64_t side = base;

    // Binary Exponentiation (Right-to-Left)
    // We use a while loop that clears the lowest set bit in each iteration.
    // This is significantly faster for sparse exponents and reduces branch pressure.
    // The __builtin_ctzll instruction finds the number of trailing zeros.
    
    while (true) {
        // If the current LSB is 1, multiply res by side
        if (exp & 1) {
            res *= side;
        }
        
        // Shift out the bit we just processed
        exp >>= 1;
        
        // If no bits are left, we are done
        if (exp == 0) break;
        
        // Square the base for the next bit position
        side *= side;
        
        // Optimization: If exp is even, we can skip the 'res' multiplication 
        // and just keep squaring 'side' until we hit the next set bit.
        if (!(exp & 1)) {
            int zeros = __builtin_ctzll(exp);
            // Manually unroll the squaring to improve ILP
            for (int i = 0; i < zeros; ++i) {
                side *= side;
            }
            exp >>= zeros;
            // After skipping zeros, the new LSB is guaranteed to be 1
            res *= side;
            if (exp == 1) break; // If that was the last bit, exit
            side *= side;
            exp >>= 1;
            if (exp == 0) break;
        }
    }

    return res;
}