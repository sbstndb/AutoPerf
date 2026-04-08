#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    // Fast paths for identity and trivial cases
    if (exp == 0) return 1;
    if (base <= 1) return base;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    // We will store powers of the base: base^(2^0), base^(2^1), base^(2^2)...
    // Since exp is uint64_t, we need at most 64 entries.
    uint64_t powers[64];
    powers[0] = base;

    // Determine how many bits we actually need to process
    // This allows us to avoid checking all 64 bits.
    int leading_zeros = __builtin_clzll(exp);
    int max_bit = 63 - leading_zeros;

    // Phase 1: Pre-calculate squares.
    // This loop has a simple dependency chain that the CPU can pipeline.
    for (int i = 1; i <= max_bit; ++i) {
        base *= base;
        powers[i] = base;
    }

    // Phase 2: Accumulate the result.
    // We start with 1 and multiply by the pre-calculated powers where the bit is set.
    // Using a local array and a loop helps the compiler use CMOV or unrolling.
    uint64_t res = 1;
    for (int i = 0; i <= max_bit; ++i) {
        uint64_t bit_set = (exp >> i) & 1;
        // Branchless selection: multiply by power if bit is 1, else multiply by 1.
        // The compiler optimizes this to a CMOV or a simple mask.
        uint64_t multiplier = bit_set ? powers[i] : 1;
        res *= multiplier;
    }

    return res;
}