#pragma once
#include <stdint.h>
#include <stdlib.h>
typedef enum BOOLEAN { CBOOL_FALSE = 0, CBOOL_TRUE = 1 } cbool_t;

typedef enum ERROR_TYPE { NullPointer, RuntimeError, ValueError } error_t;

void raise_error(error_t error_type, const char *msg);