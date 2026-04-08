#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Skip trailing zeros to reduce iterations and initialize 'res'
    // This effectively calculates base^(2^tz) as the starting point
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }
    
    uint64_t res = side;
    exp >>= (tz + 1);

    // If no bits left, we are done
    if (exp == 0) return res;

    // Main binary exponentiation loop
    // We use a pattern that encourages the compiler to use CMOV
    // and unroll to exploit Instruction Level Parallelism (ILP)
    while (exp > 0) {
        side *= side;
        uint64_t next_res = res * side;
        if (exp & 1) {
            res = next_res;
        }
        exp >>= 1;
        
        // Manual unroll step to help pipeline depth
        if (exp == 0) break;
        
        side *= side;
        next_res = res * side;
        if (exp & 1) {
            res = next_res;
        }
        exp >>= 1;
    }

    return res;
}