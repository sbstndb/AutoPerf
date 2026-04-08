uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;

    // Manual unrolling of the binary exponentiation loop.
    // This reduces the number of loop control instructions and 
    // allows the compiler to better schedule the multiplications.
    while (exp > 1) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
        
        // Secondary check to reduce loop overhead
        if (exp & 1) {
            result *= base;
        }
        if (exp > 1) {
            base *= base;
            exp >>= 1;
        }
    }
    
    return result * base;
}