#include "sequence.h"
#include <stdlib.h>

sequence_t* new_sequence() {
    sequence_t* seq = (sequence_t*)malloc(sizeof(sequence_t));
    seq->layers = NULL;
    seq->num_layers = 0;
    return seq;
}

void free_sequence(sequence_t* seq) {
    if (seq != NULL) {
        // Freeing the layers themselves should be handled outside
        if (seq->layers != NULL) {
            free(seq->layers);
        }
        free(seq);
    }
}

void sequence_add_layer(sequence_t* seq, void* layer) {
    seq->num_layers++;
    seq->layers = realloc(seq->layers, sizeof(void*) * seq->num_layers);
    seq->layers[seq->num_layers - 1] = layer;
}

// This is a simplified forward pass that assumes all layers are linear
tensor_f32_t* sequence_forward(sequence_t* seq, tensor_f32_t* src) {
    tensor_f32_t* current_input = src;
    tensor_f32_t* current_output = NULL;

    for (uint64_t i = 0; i < seq->num_layers; i++) {
        linear_layer_t* layer = (linear_layer_t*)seq->layers[i];
        current_output = linear_layer_forward(layer, current_input);

        if (current_input != src) {
            free_tensor_f32(current_input);
        }
        current_input = current_output;
    }
    return current_output;
}
