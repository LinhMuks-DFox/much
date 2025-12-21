#include "tensor.h"
#include "mnist.h"
#include "layer.h"
#include "crossentropy.h"
#include "argmax.h"
#include "optimizer.h"
#include <stdio.h>

void zero_grad(linear_layer_t* layer) {
    for (uint64_t i = 0; i < layer->weight->meta.capacity; i++) {
        layer->weight->grad[i] = 0.0f;
    }
    for (uint64_t i = 0; i < layer->bias->meta.capacity; i++) {
        layer->bias->grad[i] = 0.0f;
    }
}

int main() {
    // Load the MNIST dataset
    mnist_dataset_t* train_dataset = load_mnist_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    mnist_dataset_t* test_dataset = load_mnist_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    // Create a 3-layer neural network
    linear_layer_t* layer1 = new_linear_layer(784, 128, CBOOL_TRUE);
    linear_layer_t* layer2 = new_linear_layer(128, 64, CBOOL_TRUE);
    linear_layer_t* layer3 = new_linear_layer(64, 10, CBOOL_TRUE);

    // Create optimizers
    adam_optimizer_t* optimizer1 = new_adam_optimizer(layer1->weight->meta.capacity + layer1->bias->meta.capacity);
    adam_optimizer_t* optimizer2 = new_adam_optimizer(layer2->weight->meta.capacity + layer2->bias->meta.capacity);
    adam_optimizer_t* optimizer3 = new_adam_optimizer(layer3->weight->meta.capacity + layer3->bias->meta.capacity);

    // Training parameters
    float learning_rate = 0.001f;
    int epochs = 1;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (uint64_t i = 0; i < train_dataset->num_items; i++) {
            zero_grad(layer1);
            zero_grad(layer2);
            zero_grad(layer3);

            // Forward pass
            tensor_f32_t* out1 = linear_layer_forward(layer1, train_dataset->images[i]);
            tensor_f32_t* act1 = tensor_f32_relu(out1);
            tensor_f32_t* out2 = linear_layer_forward(layer2, act1);
            tensor_f32_t* act2 = tensor_f32_relu(out2);
            tensor_f32_t* out3 = linear_layer_forward(layer3, act2);
            
            // Calculate loss
            tensor_f32_t* loss = new_tensor_f32((uint64_t[]){1}, 1, CBOOL_TRUE);
            crossentropy_forward(loss, out3, train_dataset->labels[i]);
            total_loss += loss->data[0];

            // Backward pass
            backward(loss);
            
            // Update weights
            adam_update(optimizer1, layer1, learning_rate);
            adam_update(optimizer2, layer2, learning_rate);
            adam_update(optimizer3, layer3, learning_rate);

            free_tensor_f32(out1);
            free_tensor_f32(act1);
            free_tensor_f32(out2);
            free_tensor_f32(act2);
            free_tensor_f32(out3);
            free_tensor_f32(loss);

            if (i > 0 && i % 1000 == 0) {
                printf("Epoch %d, item %lu, loss: %.4f\n", epoch, i, total_loss / i);
            }
        }
        printf("Epoch %d, final loss: %.4f\n", epoch, total_loss / train_dataset->num_items);
    }
    
    // Test the model
    int correct = 0;
    for (uint64_t i = 0; i < test_dataset->num_items; i++) {
        tensor_f32_t* out1 = linear_layer_forward(layer1, test_dataset->images[i]);
        tensor_f32_t* act1 = tensor_f32_relu(out1);
        tensor_f32_t* out2 = linear_layer_forward(layer2, act1);
        tensor_f32_t* act2 = tensor_f32_relu(out2);
        tensor_f32_t* out3 = linear_layer_forward(layer3, act2);
        
        if (argmax(out3->data, 10) == argmax(test_dataset->labels[i]->data, 10)) {
            correct++;
        }

        free_tensor_f32(out1);
        free_tensor_f32(act1);
        free_tensor_f32(out2);
        free_tensor_f32(act2);
        free_tensor_f32(out3);
    }
    printf("Accuracy: %.2f%%\n", (float)correct / test_dataset->num_items * 100.0f);
    
    // Save the weights
    FILE* weight_file = fopen("weights.bin", "wb");
    if (weight_file) {
        fwrite(layer1->weight->data, sizeof(float), layer1->weight->meta.capacity, weight_file);
        fwrite(layer1->bias->data, sizeof(float), layer1->bias->meta.capacity, weight_file);
        fwrite(layer2->weight->data, sizeof(float), layer2->weight->meta.capacity, weight_file);
        fwrite(layer2->bias->data, sizeof(float), layer2->bias->meta.capacity, weight_file);
        fwrite(layer3->weight->data, sizeof(float), layer3->weight->meta.capacity, weight_file);
        fwrite(layer3->bias->data, sizeof(float), layer3->bias->meta.capacity, weight_file);
        fclose(weight_file);
    }
    
    // Free memory
    free_mnist_dataset(train_dataset);
    free_mnist_dataset(test_dataset);
    free_linear_layer(layer1);
    free_linear_layer(layer2);
    free_linear_layer(layer3);
    free_adam_optimizer(optimizer1);
    free_adam_optimizer(optimizer2);
    free_adam_optimizer(optimizer3);


    return 0;
}
