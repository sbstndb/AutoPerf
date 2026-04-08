#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for base 0
    if (base == 0) return exp == 0 ? 1 : 0;
    
    // Fast path for small exponents to avoid loop overhead
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Unroll the loop to process 2 bits at a time for better ILP
    // and use branchless updates to avoid mispredictions.
    while (exp > 1) {
        // Bit 0
        uint64_t mask0 = -(exp & 1);
        result *= (base & mask0) | (1 & ~mask0);
        base *= base;
        
        // Bit 1
        uint64_t mask1 = -((exp >> 1) & 1);
        result *= (base & mask1) | (1 & ~mask1);
        base *= base;

        exp >>= 2;
    }

    // Handle the final bit if exp was odd after shifts
    if (exp) {
        result *= base;
    }

    return result;
}