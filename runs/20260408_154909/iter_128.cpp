#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (exp == 1) return base;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    // Skip trailing zeros to reduce iterations
    int tz = __builtin_ctzll(exp);
    uint64_t side = base;
    for (int i = 0; i < tz; ++i) {
        side *= side;
    }
    exp >>= tz;

    uint64_t res = 1;
    
    // Main loop: Process bits without branching.
    // We use a simple while loop that the compiler can unroll.
    // The 'multiplier' logic replaces the 'if (exp & 1)' branch.
    while (exp > 1) {
        uint64_t multiplier = (exp & 1) ? side : 1;
        res *= multiplier;
        side *= side;
        exp >>= 1;
    }

    // The last bit (MSB) is always 1 because exp > 0 and we shifted.
    return res * side;
}