#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    
    // Special case for base 2 (Shift)
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
    
    // Special case for base -1 (0xFFFFFFFFFFFFFFFF)
    // Original code: (-1)^even = 1, (-1)^odd = -1
    if (base == ~0ULL) return (exp & 1) ? ~0ULL : 1ULL;

    uint64_t result = 1;

    // Skip trailing zeros in exponent to reduce iterations
    // result = 1 * (base ^ (2^ctz)) -> result = base, then square base ctz times
    int ctz = __builtin_ctzll(exp);
    exp >>= ctz;
    
    // Square the base for the trailing zeros
    while (ctz--) {
        base *= base;
    }
    
    // First '1' bit: result starts as base
    result = base;
    exp >>= 1;

    // Process remaining bits
    // We use a branchless approach where possible, but for uint64_t 
    // multiplication, the compiler usually generates optimal imul.
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
    }

    return result;
}