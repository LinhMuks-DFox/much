#include "much/layer.h"

linear_layer_t* new_linear_layer(uint64_t input_features, uint64_t output_features, cbool_t require_grad) {
    linear_layer_t* layer = (linear_layer_t*)malloc(sizeof(linear_layer_t));
    uint64_t weight_shape[] = {output_features, input_features};
    layer->weight = new_tensor_f32(weight_shape, 2, require_grad);
    uint64_t bias_shape[] = {output_features};
    layer->bias = new_tensor_f32(bias_shape, 1, require_grad);

    // Initialize weights and biases
    tensor_f32_randn(layer->weight, 0.0f, 1.0f);
    tensor_f32_fill(layer->bias, 0.0f);

    return layer;
}

void free_linear_layer(linear_layer_t* layer) {
    if (layer != NULL) {
        free_tensor_f32(layer->weight);
        free_tensor_f32(layer->bias);
        free(layer);
    }
}

tensor_f32_t* linear_layer_forward(linear_layer_t* layer, tensor_f32_t* src) {
    tensor_f32_t* matmul_out = tensor_f32_matmul(layer->weight, src);
    tensor_f32_t* add_out = tensor_f32_add(matmul_out, layer->bias);
    return add_out;
}
