#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;

    uint64_t result = 1;

    // Unroll the loop to process 2 bits at a time.
    // This reduces branch overhead and exposes ILP to the scheduler.
    while (exp > 1) {
        // Bit 0
        uint64_t next_res = result * base;
        if (exp & 1) result = next_res;
        base *= base;
        exp >>= 1;

        // Bit 1 (Unrolled)
        // We check exp > 1 in the loop header, but after the shift, 
        // we check if we still have bits to avoid unnecessary work.
        uint64_t next_res2 = result * base;
        if (exp & 1) result = next_res2;
        base *= base;
        exp >>= 1;
    }

    // Handle the final bit if it exists
    if (exp) {
        result *= base;
    }

    return result;
}