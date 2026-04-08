#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: 0^exp = 0 (if exp>0), 1^exp = 1, base^0 = 1
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Special case: Power of 2
    // 2^exp is a simple bit shift. If exp >= 64, it overflows to 0 in 64-bit unsigned.
    if (base == 2) {
        return (exp < 64) ? (1ULL << exp) : 0;
    }

    // Remove trailing zeros from exponent to reduce iterations.
    // Example: base^12 = (base^4)^3. We square base twice, then solve for exp=3.
    int tz = __builtin_ctzll(exp);
    exp >>= tz;
    while (tz--) {
        base *= base;
    }

    uint64_t res = 1;
    
    // Binary Exponentiation with Unrolling and ILP
    // We process the exponent bits. By squaring the base ahead of time,
    // we allow the CPU to execute multiple multiplications in parallel.
    while (exp > 0) {
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        
        base *= base;
        
        // Manual unroll to exploit Instruction Level Parallelism
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
        if (exp == 0) break;
        
        base *= base;
    }

    return res;
}