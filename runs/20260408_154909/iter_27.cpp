#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // Use trailing zeros to skip initial multiplications where res is just 1
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    
    // Square the base for each trailing zero
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }

    uint64_t res = side;
    exp >>= (tz + 1);

    // If no bits left, we are done
    if (exp == 0) return res;

    // Main binary exponentiation loop
    // We use a form that minimizes branching. 
    // The compiler will likely unroll this or use cmov.
    while (exp > 0) {
        side *= side;
        if (exp & 1) {
            res *= side;
        }
        exp >>= 1;
        
        // Optional: Manual unroll for the next bit to increase ILP
        if (exp == 0) break;
        
        side *= side;
        if (exp & 1) {
            res *= side;
        }
        exp >>= 1;
    }

    return res;
}