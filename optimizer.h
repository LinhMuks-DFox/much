#pragma once
#include "layer.h"

typedef struct {
    float beta1;
    float beta2;
    float epsilon;
    float* m;
    float* v;
    int t;
} adam_optimizer_t;

adam_optimizer_t* new_adam_optimizer(uint64_t num_params);
void free_adam_optimizer(adam_optimizer_t* optimizer);
void adam_update(adam_optimizer_t* optimizer, linear_layer_t* layer, float learning_rate);
