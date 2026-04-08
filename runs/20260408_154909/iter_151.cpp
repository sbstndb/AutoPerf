#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: x^0 = 1
    if (exp == 0) return 1;
    // 0^y = 0, 1^y = 1
    if (base <= 1) return base;

    // Use __builtin_clzll to find the highest set bit and start from there.
    // This avoids unnecessary iterations for small exponents.
    int leading_zeros = __builtin_clzll(exp);
    int bit_pos = 63 - leading_zeros;
    
    uint64_t res = 1;
    uint64_t b = base;

    // We process the exponent bits from least significant to most significant.
    // To maximize throughput, we use a loop that the compiler can easily 
    // unroll and optimize for the Intel Ultra 7's deep pipeline.
    while (exp > 0) {
        // If the current bit is set, multiply the result by the current base power.
        // We use a conditional assignment that maps to CMOV to avoid branch mispredictions.
        if (exp & 1) {
            res *= b;
        }
        
        // Square the base for the next bit.
        b *= b;
        exp >>= 1;

        // Early exit if exp becomes 0 to save cycles on high-order bits.
        if (exp == 0) break;

        // Manual unroll of 1 iteration to help the scheduler.
        if (exp & 1) {
            res *= b;
        }
        b *= b;
        exp >>= 1;
    }

    return res;
}