#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return 1ULL << (exp < 64 ? exp : 0); // Avoid UB, though uint64_t pow 2 usually wraps

    // Skip trailing zeros in exponent to reduce iterations
    // result = base^(exp) = (base^(2^ctz))^(exp >> ctz)
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    while (ctz--) {
        base *= base;
    }

    uint64_t result = 1;

    // Right-to-Left binary exponentiation with unrolling
    // This allows the CPU to execute base *= base and result *= base in parallel
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        
        // Unroll once to reduce branch overhead and increase ILP
        uint64_t next_bit = (exp >> 1) & 1;
        if (next_bit) {
            result *= base;
        }
        base *= base;
        
        exp >>= 2;
    }

    // Final multiply if exp was odd (or after shifts)
    if (exp) {
        result *= base;
    }

    return result;
}