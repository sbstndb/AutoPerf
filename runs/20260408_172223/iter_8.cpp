#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    
    uint64_t result = 1;

    // Unrolling the loop to reduce branch overhead and improve ILP.
    // We process bits in chunks to allow the CPU to pipeline the multiplications.
    while (exp > 0) {
        // Use conditional multiplication to avoid branch mispredictions.
        // If (exp & 1) is 0, we multiply by 1 (no-op).
        // If (exp & 1) is 1, we multiply by base.
        uint64_t mask = -(exp & 1);
        result *= ((base & mask) | (mask ^ -1ULL));
        
        base *= base;
        exp >>= 1;
        
        if (exp == 0) break;

        mask = -(exp & 1);
        result *= ((base & mask) | (mask ^ -1ULL));
        base *= base;
        exp >>= 1;
    }
    
    return result;
}