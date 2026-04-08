#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and bases
    if (exp == 0) return 1;
    if (base <= 2) {
        if (base == 2) return (exp < 64) ? (1ULL << exp) : 0;
        return base; // base 0 or 1
    }

    uint64_t res = 1;
    
    // LSB-to-MSB Binary Exponentiation (Exponentiation by Squaring)
    // This allows the CPU to execute 'base *= base' and 'res *= base' 
    // with higher instruction-level parallelism compared to MSB-to-LSB.
    while (exp > 0) {
        // If the current bit is 1, multiply the result by the current base power
        if (exp & 1) {
            res *= base;
        }
        
        // Square the base for the next bit
        base *= base;
        
        // Shift exponent to process the next bit
        exp >>= 1;
        
        // Optimization: If base becomes 0 due to overflow in 64-bit math, 
        // further multiplications won't change res (if bit is set) or base.
        // However, in standard C++, unsigned overflow is defined as wrap-around.
        // The loop will naturally terminate as exp reaches 0.
    }

    return res;
}