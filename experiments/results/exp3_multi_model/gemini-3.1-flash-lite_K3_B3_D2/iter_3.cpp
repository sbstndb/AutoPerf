uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    
    // Binary Exponentiation (Exponentiation by Squaring)
    // Reduces complexity from O(exp) to O(log exp)
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    return result;
}