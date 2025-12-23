#pragma once
#include "much/tensor.h"

void crossentropy_forward(tensor_f32_t* ret, tensor_f32_t* logits, tensor_f32_t* labels);
