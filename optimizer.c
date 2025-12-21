#include "optimizer.h"
#include <stdlib.h>
#include <math.h>

adam_optimizer_t* new_adam_optimizer(uint64_t num_params) {
    adam_optimizer_t* optimizer = (adam_optimizer_t*)malloc(sizeof(adam_optimizer_t));
    optimizer->beta1 = 0.9f;
    optimizer->beta2 = 0.999f;
    optimizer->epsilon = 1e-8f;
    optimizer->t = 0;
    optimizer->m = (float*)calloc(num_params, sizeof(float));
    optimizer->v = (float*)calloc(num_params, sizeof(float));
    return optimizer;
}

void free_adam_optimizer(adam_optimizer_t* optimizer) {
    if (optimizer != NULL) {
        free(optimizer->m);
        free(optimizer->v);
        free(optimizer);
    }
}

void adam_update(adam_optimizer_t* optimizer, linear_layer_t* layer, float learning_rate) {
    optimizer->t++;
    uint64_t param_index = 0;

    for (uint64_t i = 0; i < layer->weight->meta.capacity; i++) {
        optimizer->m[param_index] = optimizer->beta1 * optimizer->m[param_index] + (1 - optimizer->beta1) * layer->weight->grad[i];
        optimizer->v[param_index] = optimizer->beta2 * optimizer->v[param_index] + (1 - optimizer->beta2) * powf(layer->weight->grad[i], 2);
        float m_hat = optimizer->m[param_index] / (1 - powf(optimizer->beta1, optimizer->t));
        float v_hat = optimizer->v[param_index] / (1 - powf(optimizer->beta2, optimizer->t));
        layer->weight->data[i] -= learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
        param_index++;
    }

    for (uint64_t i = 0; i < layer->bias->meta.capacity; i++) {
        optimizer->m[param_index] = optimizer->beta1 * optimizer->m[param_index] + (1 - optimizer->beta1) * layer->bias->grad[i];
        optimizer->v[param_index] = optimizer->beta2 * optimizer->v[param_index] + (1 - optimizer->beta2) * powf(layer->bias->grad[i], 2);
        float m_hat = optimizer->m[param_index] / (1 - powf(optimizer->beta1, optimizer->t));
        float v_hat = optimizer->v[param_index] / (1 - powf(optimizer->beta2, optimizer->t));
        layer->bias->data[i] -= learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
        param_index++;
    }
}
