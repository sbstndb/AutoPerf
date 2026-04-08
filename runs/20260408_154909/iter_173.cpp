#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity and edge cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case for base 2: use bit shift. 
    // On x86_64, SHRX/SHL are extremely fast.
    if (base == 2) return (exp >= 64) ? 0 : (1ULL << exp);

    uint64_t res = 1;
    uint64_t b = base;

    // Binary Exponentiation (Square and Multiply)
    // We use a branchless approach for the multiplication to maximize throughput.
    // The compiler will likely optimize (exp & 1) ? b : 1 into a CMOV.
    
    while (exp > 1) {
        // Bit 0
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;

        // Bit 1 (Unrolled to improve ILP)
        if (exp & 1) res *= b;
        b *= b;
        exp >>= 1;
        
        // If exp is large, this unrolling reduces loop overhead and 
        // allows the CPU to pipeline the 'b*b' independent of 'res'.
    }

    // Final bit
    if (exp) res *= b;

    return res;
}