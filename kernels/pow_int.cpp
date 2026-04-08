#include <cstdint>

uint64_t pow_int(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    for (uint64_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}
