#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast path for common small exponents and identity cases
    if (exp == 0) return 1;
    if (exp == 1) return base;
    if (base <= 1) return base;

    // Use __builtin_ctzll to find the first set bit and initialize 'res'
    // This avoids the 'res = 1' multiplication in the first iteration.
    int ct = __builtin_ctzll(exp);
    uint64_t res = base;
    
    // Square the base 'ct' times to account for trailing zeros
    // This is unrolled manually to maximize ILP
    for (int i = 0; i < ct; ++i) {
        res *= res;
    }

    uint64_t current_base = res;
    exp >>= (ct + 1);

    // If no bits are left, we are done
    if (exp == 0) return res;

    // Standard binary exponentiation for remaining bits
    // The loop is structured to minimize branching
    while (exp > 0) {
        current_base *= current_base;
        if (exp & 1) {
            res *= current_base;
        }
        exp >>= 1;
        
        // Optimization: if exp is small, we can finish early
        // This helps with short-circuiting the loop for small exponents
        if (exp == 0) break;

        current_base *= current_base;
        if (exp & 1) {
            res *= current_base;
        }
        exp >>= 1;
    }

    return res;
}