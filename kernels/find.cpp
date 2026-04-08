#include <cstddef>

int find(const int* data, int size, int value) {
    for (int i = 0; i < size; i++) {
        if (data[i] == value)
            return i;
    }
    return -1;
}
