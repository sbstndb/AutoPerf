#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Handle the most common/trivial cases immediately
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;
    if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;

    uint64_t res = 1;

    // Unroll the first 3 iterations to exploit Instruction Level Parallelism (ILP).
    // This allows the CPU to pre-calculate higher powers of 'base' 
    // while the 'res' multiplications are still pending.
    
    // Bit 0
    if (exp & 1) res = base;
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;

    // Bit 1
    if (exp & 1) res *= base;
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;

    // Bit 2
    if (exp & 1) res *= base;
    exp >>= 1;
    if (exp == 0) return res;
    base *= base;

    // Remaining bits: Standard loop for larger exponents.
    // At this point, exp is likely small, so the loop runs few times.
    while (exp > 0) {
        if (exp & 1) res *= base;
        
        // Check if base will overflow/become irrelevant 
        // (Optional, but helps if exp is very large)
        base *= base;
        exp >>= 1;
    }

    return res;
}