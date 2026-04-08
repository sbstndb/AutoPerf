#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Identity cases: handle 0^0 as 1, x^0 as 1
    if (exp == 0) return 1;
    // 0^y = 0, 1^y = 1
    if (base <= 1) return base;

    // Skip trailing zeros in exponent to initialize 'res'
    // This reduces the number of multiplications and avoids 
    // multiplying res by 1 in the first iteration.
    int trailing_zeros = __builtin_ctzll(exp);
    exp >>= trailing_zeros;
    
    // Square the base for every trailing zero bit
    // This is faster than starting res=1 and checking bits
    uint64_t b = base;
    for (int i = 0; i < trailing_zeros; ++i) {
        b *= b;
    }

    uint64_t res = b;
    exp >>= 1;

    // Standard Binary Exponentiation (Bottom-up)
    // We use a branchless approach for the multiplication.
    // Modern CPUs (like Ultra 7) execute the ternary as a CMOV.
    while (exp > 0) {
        b *= b;
        uint64_t next_res = res * b;
        // If the lowest bit is 1, update res, otherwise keep it.
        res = (exp & 1) ? next_res : res;
        exp >>= 1;
    }

    return res;
}