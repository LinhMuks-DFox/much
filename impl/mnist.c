#include "much/mnist.h"
#include <stdio.h>
#include <stdlib.h>

int32_t bswap_32(int32_t val) {
    return ((val & 0xFF) << 24) |
           ((val & 0xFF00) << 8) |
           ((val >> 8) & 0xFF00) |
           ((val >> 24) & 0xFF);
}

mnist_dataset_t* load_mnist_dataset(const char* image_path, const char* label_path) {
    FILE* image_file = fopen(image_path, "rb");
    if (!image_file) {
        raise_error(RuntimeError, "Could not open image file");
    }

    FILE* label_file = fopen(label_path, "rb");
    if (!label_file) {
        raise_error(RuntimeError, "Could not open label file");
    }

    int32_t magic, num_images, num_labels, rows, cols;

    fread(&magic, sizeof(int32_t), 1, image_file);
    magic = bswap_32(magic);
    if (magic != 2051) {
        raise_error(RuntimeError, "Invalid magic number in image file");
    }
    
    fread(&num_images, sizeof(int32_t), 1, image_file);
    num_images = bswap_32(num_images);

    fread(&rows, sizeof(int32_t), 1, image_file);
    rows = bswap_32(rows);

    fread(&cols, sizeof(int32_t), 1, image_file);
    cols = bswap_32(cols);

    fread(&magic, sizeof(int32_t), 1, label_file);
    magic = bswap_32(magic);
    if (magic != 2049) {
        raise_error(RuntimeError, "Invalid magic number in label file");
    }

    fread(&num_labels, sizeof(int32_t), 1, label_file);
    num_labels = bswap_32(num_labels);
    
    if (num_images != num_labels) {
        raise_error(ValueError, "Number of images and labels do not match");
    }

    mnist_dataset_t* dataset = (mnist_dataset_t*)malloc(sizeof(mnist_dataset_t));
    dataset->num_items = num_images;
    dataset->images = (tensor_f32_t**)malloc(sizeof(tensor_f32_t*) * num_images);
    dataset->labels = (tensor_f32_t**)malloc(sizeof(tensor_f32_t*) * num_images);

    uint64_t image_shape[] = {rows * cols, 1};
    uint64_t label_shape[] = {10, 1};

    for (int i = 0; i < num_images; i++) {
        dataset->images[i] = new_tensor_f32(image_shape, 2, CBOOL_FALSE);
        dataset->labels[i] = new_tensor_f32(label_shape, 2, CBOOL_FALSE);

        uint8_t* image_data = (uint8_t*)malloc(rows * cols);
        fread(image_data, 1, rows * cols, image_file);
        for (int j = 0; j < rows * cols; j++) {
            dataset->images[i]->data[j] = (float)image_data[j] / 255.0f;
        }
        free(image_data);

        uint8_t label_data;
        fread(&label_data, 1, 1, label_file);
        for (int j = 0; j < 10; j++) {
            dataset->labels[i]->data[j] = (j == label_data) ? 1.0f : 0.0f;
        }
    }

    fclose(image_file);
    fclose(label_file);

    return dataset;
}

void free_mnist_dataset(mnist_dataset_t* dataset) {
    if (dataset != NULL) {
        for (uint64_t i = 0; i < dataset->num_items; i++) {
            free_tensor_f32(dataset->images[i]);
            free_tensor_f32(dataset->labels[i]);
        }
        free(dataset->images);
        free(dataset->labels);
        free(dataset);
    }
}
