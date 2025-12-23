#pragma once
#include "much/tensor.h"

typedef struct {
    tensor_f32_t** images;
    tensor_f32_t** labels;
    uint64_t num_items;
} mnist_dataset_t;

mnist_dataset_t* load_mnist_dataset(const char* image_path, const char* label_path);
void free_mnist_dataset(mnist_dataset_t* dataset);
