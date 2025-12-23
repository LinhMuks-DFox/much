#include "much/tensor.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t tensor_alloc_count = 0;

uint64_t get_tensor_alloc_count() { return tensor_alloc_count; }

void init_tensor_meta(tensor_meta *self, uint64_t capacity, uint64_t *shape,
                      uint64_t shape_length, cbool_t require_grad) {
  if (self == NULL) {
    raise_error(NullPointer, "tensor_meta* self is NULL");
  }

  if (shape == NULL) {
    raise_error(NullPointer, "uint64_t* shape is NULL");
  }
  self->capacity = capacity;
  self->shape_length = shape_length;
  self->require_grad = require_grad;
  self->shape = (uint64_t *)malloc(sizeof(uint64_t) * shape_length);
  if (self->shape == NULL) {
    raise_error(NullPointer, "malloc failed to allocate shape");
  }
  memcpy(self->shape, shape, shape_length * sizeof(uint64_t));
}

tensor_meta *new_tensor_meta(uint64_t capacity, uint64_t *shape,
                             uint64_t shape_length, cbool_t require_grad) {
  tensor_meta *ret = (tensor_meta *)malloc(sizeof(tensor_meta));
  init_tensor_meta(ret, capacity, shape, shape_length, require_grad);
  return ret;
}

void free_tensor_meta(tensor_meta *self) {
  if (self != NULL) {
    if (self->shape != NULL) {
      free(self->shape);
    }
    free(self);
  }
}

tensor_f32_t *new_tensor_f32(uint64_t *shape, uint64_t shape_length,
                             cbool_t require_grad) {
  tensor_f32_t *ret = (tensor_f32_t *)malloc(sizeof(tensor_f32_t));
  if (ret == NULL) {
    raise_error(NullPointer, "malloc failed to allocate tensor_f32_t");
  }

  uint64_t capacity = 1;
  for (uint64_t i = 0; i < shape_length; i++) {
    capacity *= shape[i];
  }

  init_tensor_meta(&ret->meta, capacity, shape, shape_length, require_grad);

  ret->data = (float *)malloc(sizeof(float) * capacity);
  if (ret->data == NULL) {
    raise_error(NullPointer, "malloc failed to allocate tensor data");
  }

  if (require_grad == CBOOL_TRUE) {
    ret->grad = (float *)calloc(capacity, sizeof(float));
    if (ret->grad == NULL) {
      raise_error(NullPointer, "malloc failed to allocate tensor grad");
    }
  } else {
    ret->grad = NULL;
  }

  ret->backward_fn = NULL;
  ret->prev = NULL;
  ret->num_prev = 0;

  tensor_alloc_count++;

  return ret;
}

void free_tensor_f32(tensor_f32_t *self) {
  if (self != NULL) {
    if (self->data != NULL) {
      free(self->data);
    }
    if (self->grad != NULL) {
      free(self->grad);
    }
    if (self->meta.shape != NULL) {
      free(self->meta.shape);
    }
    if (self->prev != NULL) {
      free(self->prev);
    }
    free(self);
    tensor_alloc_count--;
  }
}

void tensor_f32_fill(tensor_f32_t *self, float value) {
  if (self == NULL || self->data == NULL) {
    raise_error(NullPointer, "tensor or tensor data is NULL");
  }
  for (uint64_t i = 0; i < self->meta.capacity; i++) {
    self->data[i] = value;
  }
}

// Box-Muller transform
void tensor_f32_randn(tensor_f32_t *self, float mean, float std) {
  if (self == NULL || self->data == NULL) {
    raise_error(NullPointer, "tensor or tensor data is NULL");
  }
  for (uint64_t i = 0; i < self->meta.capacity; i += 2) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);
    self->data[i] = z1 * std + mean;
    if (i + 1 < self->meta.capacity) {
      self->data[i + 1] = z2 * std + mean;
    }
  }
}

// Backward functions
void add_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  tensor_f32_t *b = self->prev[1];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      a->grad[i] += self->grad[i];
    }
  }
  if (b->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < b->meta.capacity; i++) {
      b->grad[i] += self->grad[i];
    }
  }
}

void sub_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  tensor_f32_t *b = self->prev[1];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      a->grad[i] += self->grad[i];
    }
  }
  if (b->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < b->meta.capacity; i++) {
      b->grad[i] -= self->grad[i];
    }
  }
}

void mul_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  tensor_f32_t *b = self->prev[1];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      a->grad[i] += self->grad[i] * b->data[i];
    }
  }
  if (b->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < b->meta.capacity; i++) {
      b->grad[i] += self->grad[i] * a->data[i];
    }
  }
}

void div_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  tensor_f32_t *b = self->prev[1];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      a->grad[i] += self->grad[i] / b->data[i];
    }
  }
  if (b->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < b->meta.capacity; i++) {
      b->grad[i] -= self->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
    }
  }
}

void matmul_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  tensor_f32_t *b = self->prev[1];
  if (a->meta.require_grad == CBOOL_TRUE) {
    // a->grad += self->grad * b^T
    uint64_t M = self->meta.shape[0];
    uint64_t N = b->meta.shape[0];
    uint64_t K = b->meta.shape[1];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f,
                self->grad, K, b->data, K, 1.0f, a->grad, N);
  }
  if (b->meta.require_grad == CBOOL_TRUE) {
    // b->grad += a^T * self->grad
    uint64_t M = a->meta.shape[1];
    uint64_t N = self->meta.shape[1];
    uint64_t K = a->meta.shape[0];
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, a->data,
                M, self->grad, N, 1.0f, b->grad, N);
  }
}

void sigmoid_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      float s = self->data[i];
      a->grad[i] += self->grad[i] * s * (1 - s);
    }
  }
}

void relu_backward(tensor_f32_t *self) {
  tensor_f32_t *a = self->prev[0];
  if (a->meta.require_grad == CBOOL_TRUE) {
    for (uint64_t i = 0; i < a->meta.capacity; i++) {
      if (a->data[i] > 0) {
        a->grad[i] += self->grad[i];
      } else {
        a->grad[i] += self->grad[i] * 0.01f;
      }
    }
  }
}

tensor_f32_t *tensor_f32_add(tensor_f32_t *a, tensor_f32_t *b) {
  if (a->meta.capacity != b->meta.capacity) {
    raise_error(ValueError, "tensor shapes are not compatible for addition");
  }
  cbool_t require_grad =
      a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    ret->data[i] = a->data[i] + b->data[i];
  }
  if (require_grad) {
    ret->backward_fn = add_backward;
    ret->num_prev = 2;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *) * ret->num_prev);
    ret->prev[0] = a;
    ret->prev[1] = b;
  }
  return ret;
}

tensor_f32_t *tensor_f32_sub(tensor_f32_t *a, tensor_f32_t *b) {
  if (a->meta.capacity != b->meta.capacity) {
    raise_error(ValueError, "tensor shapes are not compatible for subtraction");
  }
  cbool_t require_grad =
      a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    ret->data[i] = a->data[i] - b->data[i];
  }
  if (require_grad) {
    ret->backward_fn = sub_backward;
    ret->num_prev = 2;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *) * ret->num_prev);
    ret->prev[0] = a;
    ret->prev[1] = b;
  }
  return ret;
}

tensor_f32_t *tensor_f32_mul(tensor_f32_t *a, tensor_f32_t *b) {
  if (a->meta.capacity != b->meta.capacity) {
    raise_error(ValueError,
                "tensor shapes are not compatible for multiplication");
  }
  cbool_t require_grad =
      a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    ret->data[i] = a->data[i] * b->data[i];
  }
  if (require_grad) {
    ret->backward_fn = mul_backward;
    ret->num_prev = 2;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *) * ret->num_prev);
    ret->prev[0] = a;
    ret->prev[1] = b;
  }
  return ret;
}

tensor_f32_t *tensor_f32_div(tensor_f32_t *a, tensor_f32_t *b) {
  if (a->meta.capacity != b->meta.capacity) {
    raise_error(ValueError, "tensor shapes are not compatible for division");
  }
  cbool_t require_grad =
      a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    if (b->data[i] == 0.0f) {
      raise_error(ValueError, "division by zero");
    }
    ret->data[i] = a->data[i] / b->data[i];
  }
  if (require_grad) {
    ret->backward_fn = div_backward;
    ret->num_prev = 2;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *) * ret->num_prev);
    ret->prev[0] = a;
    ret->prev[1] = b;
  }
  return ret;
}

tensor_f32_t *tensor_f32_matmul(tensor_f32_t *a, tensor_f32_t *b) {
  if (a->meta.shape_length != 2 || b->meta.shape_length != 2) {
    raise_error(ValueError, "matmul requires 2D tensors");
  }
  if (a->meta.shape[1] != b->meta.shape[0]) {
    raise_error(ValueError, "tensor shapes are not compatible for matmul");
  }

  cbool_t require_grad =
      a->meta.require_grad == CBOOL_TRUE || b->meta.require_grad == CBOOL_TRUE;
  uint64_t ret_shape[] = {a->meta.shape[0], b->meta.shape[1]};
  tensor_f32_t *ret = new_tensor_f32(ret_shape, 2, require_grad);

  uint64_t M = a->meta.shape[0];
  uint64_t N = b->meta.shape[1];
  uint64_t K = a->meta.shape[1];

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a->data,
              K, b->data, N, 0.0f, ret->data, N);

  if (require_grad) {
    ret->backward_fn = matmul_backward;
    ret->num_prev = 2;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *) * ret->num_prev);
    ret->prev[0] = a;
    ret->prev[1] = b;
  }
  return ret;
}

tensor_f32_t *tensor_f32_sigmoid(tensor_f32_t *a) {
  cbool_t require_grad = a->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    ret->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
  }
  if (require_grad) {
    ret->backward_fn = sigmoid_backward;
    ret->num_prev = 1;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *));
    ret->prev[0] = a;
  }
  return ret;
}

tensor_f32_t *tensor_f32_relu(tensor_f32_t *a) {
  cbool_t require_grad = a->meta.require_grad == CBOOL_TRUE;
  tensor_f32_t *ret =
      new_tensor_f32(a->meta.shape, a->meta.shape_length, require_grad);
  for (uint64_t i = 0; i < a->meta.capacity; i++) {
    ret->data[i] = a->data[i] > 0 ? a->data[i] : a->data[i] * 0.01f;
  }
  if (require_grad) {
    ret->backward_fn = relu_backward;
    ret->num_prev = 1;
    ret->prev = (tensor_f32_t **)malloc(sizeof(tensor_f32_t *));
    ret->prev[0] = a;
  }
  return ret;
}

static cbool_t tensor_list_contains(tensor_f32_t *node, tensor_f32_t **list,
                                    int list_size) {
  for (int i = 0; i < list_size; i++) {
    if (list[i] == node) {
      return CBOOL_TRUE;
    }
  }
  return CBOOL_FALSE;
}

static void tensor_list_append(tensor_f32_t *node, tensor_f32_t ***list,
                               int *list_size) {
  tensor_f32_t **new_list = (tensor_f32_t **)realloc(
      *list, sizeof(tensor_f32_t *) * (*list_size + 1));
  if (new_list == NULL) {
    raise_error(NullPointer, "realloc failed while building graph");
  }
  new_list[*list_size] = node;
  *list = new_list;
  *list_size += 1;
}

static void build_graph_dfs(tensor_f32_t *node, tensor_f32_t ***graph,
                            int *graph_size, tensor_f32_t ***visited,
                            int *visited_size) {
  if (node == NULL) {
    return;
  }

  if (tensor_list_contains(node, *visited, *visited_size) == CBOOL_TRUE) {
    return;
  }

  tensor_list_append(node, visited, visited_size);

  for (int i = 0; i < node->num_prev; i++) {
    build_graph_dfs(node->prev[i], graph, graph_size, visited, visited_size);
  }

  tensor_list_append(node, graph, graph_size);
}

void backward(tensor_f32_t *self) {
  if (self->meta.require_grad != CBOOL_TRUE) {
    raise_error(ValueError,
                "Cannot call backward on a tensor that does not require grad");
  }

  // Fill grad with 1s
  for (uint64_t i = 0; i < self->meta.capacity; i++) {
    self->grad[i] = 1.0f;
  }

  // Build the graph
  tensor_f32_t **graph = NULL;
  int graph_size = 0;
  tensor_f32_t **visited = NULL;
  int visited_size = 0;

  build_graph_dfs(self, &graph, &graph_size, &visited, &visited_size);

  // Backward pass
  for (int i = graph_size - 1; i >= 0; i--) {
    if (graph[i]->backward_fn != NULL) {
      graph[i]->backward_fn(graph[i]);
    }
  }

  free(graph);
  free(visited);
}

void print_tensor(tensor_f32_t *self) {
  if (self == NULL) {
    printf("NULL tensor\n");
    return;
  }
  printf("Tensor @ %p\n", self);
  printf("  meta:\n");
  printf("    capacity: %llu\n", (unsigned long long)self->meta.capacity);
  printf("    shape: [");
  for (uint64_t i = 0; i < self->meta.shape_length; i++) {
    printf("%llu", (unsigned long long)self->meta.shape[i]);
    if (i < self->meta.shape_length - 1) {
      printf(", ");
    }
  }
  printf("]\n");
  printf("    require_grad: %s\n",
         self->meta.require_grad == CBOOL_TRUE ? "True" : "False");
  printf("  data: [");

  // Only print first 10 elements for brevity
  uint64_t limit = self->meta.capacity > 10 ? 10 : self->meta.capacity;
  for (uint64_t i = 0; i < limit; i++) {
    printf("%.4f", self->data[i]);
    if (i < limit - 1) {
      printf(", ");
    }
  }
  if (self->meta.capacity > 10) {
    printf(", ...");
  }
  printf("]\n");

  if (self->meta.require_grad == CBOOL_TRUE) {
    printf("  grad: [");
    // Only print first 10 elements for brevity
    for (uint64_t i = 0; i < limit; i++) {
      printf("%.4f", self->grad[i]);
      if (i < limit - 1) {
        printf(", ");
      }
    }
    if (self->meta.capacity > 10) {
      printf(", ...");
    }
    printf("]\n");
  }
}
