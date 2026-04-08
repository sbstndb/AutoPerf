#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity cases to exit early
    if (exp == 0) return 1;
    if (base <= 1) return base;
    
    // Handle the first bit to initialize 'res' and avoid an extra multiply by 1
    // This also handles exp == 1 automatically.
    while ((exp & 1) == 0) {
        base *= base;
        exp >>= 1;
    }
    
    uint64_t res = base;
    exp >>= 1;

    // Standard Binary Exponentiation (Square-and-Multiply)
    // The loop is structured to allow the CPU to execute base *= base 
    // and the conditional res *= base in parallel (ILP).
    while (exp > 0) {
        base *= base;
        if (exp & 1) {
            res *= base;
        }
        exp >>= 1;
    }

    return res;
}