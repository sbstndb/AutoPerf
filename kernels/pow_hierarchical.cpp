#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;
    uint64_t half = pow_int(base * base, exp >> 1);
    return (exp & 1) ? base * half : half;
}
