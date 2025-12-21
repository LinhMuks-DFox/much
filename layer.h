#pragma once
#include "tensor.h"

typedef struct {
    tensor_f32_t* weight;
    tensor_f32_t* bias;
} linear_layer_t;

linear_layer_t* new_linear_layer(uint64_t input_features, uint64_t output_features, cbool_t require_grad);
void free_linear_layer(linear_layer_t* layer);
tensor_f32_t* linear_layer_forward(linear_layer_t* layer, tensor_f32_t* src);