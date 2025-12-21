# much - A Toy Deep Learning Framework in C

`much` is a small, educational deep learning framework written from scratch in C. It is designed to be simple and easy to understand, while still being powerful enough to train a simple neural network on the MNIST dataset.

## Features

*   **Dynamic Computation Graph:** `much` builds a dynamic computation graph, allowing for flexibility in network architecture.
*   **Automatic Differentiation:** The framework can automatically compute gradients using backpropagation.
*   **Common Layers and Optimizers:** `much` includes implementations of common layers like Linear, ReLU, and Sigmoid, as well as the Adam optimizer.
*   **MNIST Demo:** The included demo trains a 3-layer neural network on the MNIST dataset, achieving over 90% accuracy.
*   **OpenBLAS Integration:** `much` uses OpenBLAS for efficient matrix operations.

## Getting Started

### Prerequisites

*   A C compiler (like `gcc` or `clang`)
*   CMake
*   Ninja (or Make)
*   OpenBLAS

### Building and Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/much.git
    cd much
    ```

2.  **Download the MNIST dataset:**
    ```bash
    wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz \
         https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz \
         https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz \
         https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
    gunzip *.gz
    ```

3.  **Build the project:**
    ```bash
    cmake -B build -S . -G Ninja
    ninja -C build
    ```

4.  **Run the MNIST demo:**
    ```bash
    ./run_mnist.sh
    ```

## Architecture

The framework is built around a few core components:

*   **Tensor:** The fundamental data structure in `much`. It is a multi-dimensional array that can store data and gradients.
*   **Layer:** A neural network layer, such as a linear layer or an activation function.
*   **Optimizer:** An algorithm for updating the weights of the network, such as Adam.
*   **Sequence:** A container for a sequence of layers, which makes it easy to build and train a network.

The computation graph is built dynamically as operations are performed on tensors. Each tensor stores a pointer to the function that created it (the `backward_fn`), and the tensors that were used to create it (the `prev` tensors). When `backward()` is called on a tensor, it traverses the graph in reverse order, calling the `backward_fn` of each tensor to compute the gradients.

## Future Work

*   **Memory Management:** The current memory management is manual and can be improved with a memory pool or a garbage collector.
*   **More Layers and Optimizers:** The framework could be extended with more layers (like Convolutional and Recurrent layers) and optimizers (like SGD with momentum).
*   **GPU Support:** Adding GPU support would significantly speed up training.
*   **Serialization:** The ability to save and load entire models would be a useful feature.
