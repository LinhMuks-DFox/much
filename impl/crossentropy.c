#include "much/crossentropy.h"
#include <math.h>

void softmax(tensor_f32_t* out, tensor_f32_t* in) {
    float max_val = in->data[0];
    for (uint64_t i = 1; i < in->meta.capacity; i++) {
        if (in->data[i] > max_val) {
            max_val = in->data[i];
        }
    }

    float sum = 0.0f;
    for (uint64_t i = 0; i < in->meta.capacity; i++) {
        out->data[i] = expf(in->data[i] - max_val);
        sum += out->data[i];
    }

    for (uint64_t i = 0; i < in->meta.capacity; i++) {
        out->data[i] /= sum;
    }
}


void crossentropy_backward(tensor_f32_t *self) {
    tensor_f32_t *logits = self->prev[0];
    tensor_f32_t *labels = self->prev[1];
    
    if (logits->meta.require_grad == CBOOL_TRUE) {
        tensor_f32_t* softmax_out = new_tensor_f32(logits->meta.shape, logits->meta.shape_length, CBOOL_FALSE);
        softmax(softmax_out, logits);

        for (uint64_t i = 0; i < logits->meta.capacity; i++) {
            logits->grad[i] += self->grad[0] * (softmax_out->data[i] - labels->data[i]);
        }

        free_tensor_f32(softmax_out);
    }
}

void crossentropy_forward(tensor_f32_t* ret, tensor_f32_t* logits, tensor_f32_t* labels) {
    tensor_f32_t* softmax_out = new_tensor_f32(logits->meta.shape, logits->meta.shape_length, CBOOL_FALSE);
    softmax(softmax_out, logits);

    float loss = 0.0f;
    for (uint64_t i = 0; i < logits->meta.capacity; i++) {
        loss -= labels->data[i] * logf(softmax_out->data[i] + 1e-9);
    }
    ret->data[0] = loss;

    if (logits->meta.require_grad == CBOOL_TRUE) {
        ret->backward_fn = crossentropy_backward;
        ret->num_prev = 2;
        ret->prev = (tensor_f32_t**)malloc(sizeof(tensor_f32_t*) * 2);
        ret->prev[0] = logits;
        ret->prev[1] = labels;
    }
    
    free_tensor_f32(softmax_out);
}
