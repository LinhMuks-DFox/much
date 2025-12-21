#include "mse.h"
#include <math.h>

void mse_backward(tensor_f32_t *self) {
    tensor_f32_t *a = self->prev[0];
    tensor_f32_t *b = self->prev[1];
    if (a->meta.require_grad == CBOOL_TRUE) {
        for (uint64_t i = 0; i < a->meta.capacity; i++) {
            a->grad[i] += self->grad[i] * 2.0f * (a->data[i] - b->data[i]) / a->meta.capacity;
        }
    }
    if (b->meta.require_grad == CBOOL_TRUE) {
        for (uint64_t i = 0; i < b->meta.capacity; i++) {
            b->grad[i] += self->grad[i] * -2.0f * (a->data[i] - b->data[i]) / a->meta.capacity;
        }
    }
}

tensor_f32_t* mse_forward(tensor_f32_t* a, tensor_f32_t* b) {
    if (a->meta.capacity != b->meta.capacity) {
        raise_error(ValueError, "tensor shapes are not compatible for mse");
    }
    cbool_t require_grad = a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
    uint64_t ret_shape[] = {1};
    tensor_f32_t* ret = new_tensor_f32(ret_shape, 1, require_grad);

    float sum = 0.0f;
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
        sum += powf(a->data[i] - b->data[i], 2);
    }
    ret->data[0] = sum / a->meta.capacity;

    if (require_grad) {
        ret->backward_fn = mse_backward;
        ret->num_prev = 2;
        ret->prev = (tensor_f32_t**)malloc(sizeof(tensor_f32_t*) * ret->num_prev);
        ret->prev[0] = a;
        ret->prev[1] = b;
    }
    return ret;
}