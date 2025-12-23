#pragma once

#include "much/layer.h"
#include <stdint.h>

typedef struct {
    void** layers;
    uint64_t num_layers;
    // Layer types can be stored in another array if needed for forward pass
} sequence_t;

sequence_t* new_sequence();
void free_sequence(sequence_t* seq);
void sequence_add_layer(sequence_t* seq, void* layer);
tensor_f32_t* sequence_forward(sequence_t* seq, tensor_f32_t* src);
