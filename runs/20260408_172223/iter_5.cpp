uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (base == 0) return 0;
    if (base == 1) return 1;

    uint64_t result = 1;
    
    // Unrolling the loop manually for 64-bit integers.
    // We use a branchless approach to update the result.
    // The compiler will use CMOV instructions to handle the conditional multiplication.
    while (exp > 0) {
        // If exp is odd, multiply result by base.
        // Using a ternary or conditional move is faster than a branch.
        uint64_t mask = -(exp & 1);
        result *= (1 | (mask & (base - 1)));
        
        exp >>= 1;
        if (exp == 0) break;
        
        base *= base;
    }
    
    return result;
}