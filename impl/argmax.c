#include "much/argmax.h"

uint64_t argmax(float* data, uint64_t len) {
    if (len == 0) {
        return 0;
    }
    uint64_t max_index = 0;
    for (uint64_t i = 1; i < len; i++) {
        if (data[i] > data[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}
