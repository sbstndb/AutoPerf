uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;

    uint64_t result = 1;

    // Process bits in pairs to reduce loop overhead and branch frequency
    while (exp >= 4) {
        if (exp & 1) result *= base;
        base *= base;
        if (exp & 2) result *= base;
        base *= base;
        exp >>= 2;
    }

    // Handle remaining 0-3 bits
    if (exp & 1) result *= base;
    if (exp & 2) {
        base *= base;
        result *= base;
    }

    return result;
}