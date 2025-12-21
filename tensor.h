#pragma once

#include "util.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct TENSOR_META {
  uint64_t capacity;
  uint64_t *shape;
  uint64_t shape_length;
  cbool_t require_grad;
} tensor_meta;

struct FLOAT_TESNOR;

typedef void (*grad_fn)(struct FLOAT_TESNOR *self);

typedef struct FLOAT_TESNOR {
  float *data;
  float *grad;
  tensor_meta meta;

  grad_fn backward_fn;
  struct FLOAT_TESNOR** prev;
  int num_prev;
} tensor_f32_t;

tensor_meta *new_tensor_meta(uint64_t capacity, uint64_t *shape,
                             uint64_t shape_length, cbool_t require_grad);

void init_tensor_meta(tensor_meta *self, uint64_t capacity, uint64_t *shape,
                      uint64_t shape_length, cbool_t require_grad);

void free_tensor_meta(tensor_meta *self);

tensor_f32_t *new_tensor_f32(uint64_t *shape, uint64_t shape_length,
                             cbool_t require_grad);

void free_tensor_f32(tensor_f32_t *self);

void tensor_f32_fill(tensor_f32_t *self, float value);

void tensor_f32_randn(tensor_f32_t *self, float mean, float std);

tensor_f32_t* tensor_f32_add(tensor_f32_t *a, tensor_f32_t *b);
tensor_f32_t* tensor_f32_sub(tensor_f32_t *a, tensor_f32_t *b);
tensor_f32_t* tensor_f32_mul(tensor_f32_t *a, tensor_f32_t *b);
tensor_f32_t* tensor_f32_div(tensor_f32_t *a, tensor_f32_t *b);
tensor_f32_t* tensor_f32_matmul(tensor_f32_t *a, tensor_f32_t *b);
tensor_f32_t* tensor_f32_sigmoid(tensor_f32_t *a);
tensor_f32_t* tensor_f32_relu(tensor_f32_t *a);

void print_tensor(tensor_f32_t *self);

uint64_t get_tensor_alloc_count();

void backward(tensor_f32_t *self);